// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/moe/qmoe.h"
#include "contrib_ops/cpu/quantization/moe_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class GateProgram final : public Program<GateProgram> {
 public:
  GateProgram(int k, bool is_fp16) :
  Program<GateProgram>{"Gate"}, k_{k}, is_fp16_{is_fp16} {};

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("topk_values");
    shader.AddOutput("topk_indices");

    shader.AdditionalImplementation()
        << "const K: u32 = " << k_ << "4u;\n"
        << "const MAX_FLOAT = " << ((is_fp16_) ? "3.402823466e+38;\n" : "65504.0;\n")
        << "var<workgroup> shared_vals: array<input_element_t, workgroup_size_x>;\n"
        << "var<workgroup> shared_idxs: array<u32, workgroup_size_x>;\n";

    shader.MainFunctionBody() << R"TOPKSOFTMAX(

          let row = workgroup_id.x;
          if (row >= uniforms.rows) { return; }
          let tid = local_id.x;
          let cols = uniforms.cols;
          let base = row * cols;

          var max_val: input_element_t = -MAX_FLOAT;
          var max_idx: u32 = 0u;

          if (tid < cols) {
              max_val = input[base + tid];
              max_idx = tid;
          }
          shared_vals[tid] = max_val;
          shared_idxs[tid] = max_idx;
          workgroupBarrier();

          // Bitonic Top-K sorting algorithm
          for (var size = 2u; size <= workgroup_size_x; size *= 2u) {
              for (var stride = size / 2u; stride > 0u; stride /= 2u) {
                  let partner = tid ^ stride;
                  if (partner > tid && partner < workgroup_size_x && tid < cols && partner < cols) {
                      let ascending_block = ((tid & size) != 0u);
                      let should_swap = select(
                          shared_vals[partner] > shared_vals[tid],
                          shared_vals[partner] < shared_vals[tid],
                          ascending_block
                      );
                      if (should_swap) {
                          // Swap values and indices
                          let temp_val = shared_vals[tid];
                          let temp_idx = shared_idxs[tid];
                          shared_vals[tid] = shared_vals[partner];
                          shared_idxs[tid] = shared_idxs[partner];
                          shared_vals[partner] = temp_val;
                          shared_idxs[partner] = temp_idx;
                      }
                  }
                  workgroupBarrier();
              }
          }
          if (tid < K) {
              let output_base = row * K;
              let src_idx = tid;
              if (src_idx < cols) {
                  topk_indices[output_base + tid] = shared_idxs[src_idx];
              }
          }
          workgroupBarrier();
          if (tid == 0u) {
              let output_base = row * K;
              var sum : f32 = 0.0;
              for (var i = 0u; i < K; i++) {
                  sum += exp(shared_vals[i]);
              }
              for (var i = 0u; i < K; i++) {
                  topk_values[output_base + i] = input_element_t(exp(f32(shared_vals[i])) / sum);
              }
          }
      }
  )TOPKSOFTMAX";

    return Status::OK();
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
    {"rows", ProgramUniformVariableDataType::Uint32},
    {"cols", ProgramUniformVariableDataType::Uint32}
  );

 private:
  int k_;
  bool is_fp16_;
};


Status QMoEProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);

  return Status::OK();
}

Status QMoE::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input<Tensor>(0);
  const Tensor* router_probs = context.Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context.Input<Tensor>(2);
  const Tensor* fc1_scales = context.Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context.Input<Tensor>(4);
  const Tensor* fc2_experts_weights = context.Input<Tensor>(5);
  const Tensor* fc2_scales = context.Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = context.Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = context.Input<Tensor>(8);
  const Tensor* fc3_scales_optional = context.Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = context.Input<Tensor>(10);

  MoEQuantType quant_type = expert_weight_bits_ == 4 ? MoEQuantType::UINT4 : MoEQuantType::UINT8;
  MoEParameters moe_params;

  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, fc1_scales,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional,
      expert_weight_bits_ == 4 ? 2 : 1,
      activation_type_ == MoEActivationType::SwiGLU));

  /*
     export_ids, router_weights = Gate(input, router_probs)
     for expert_id in export_ids:
       # nbitmm with expert_id[] passed in via input and index passed in via uniform?
       fc1_output = FC1(input, expert_id)
       fc1_output = Activation(fc1_output + fc1_bias)
       fc2_output = FC2(fc1_output, expert_id)
       output = output * fc2_output
   */

  const auto& input_shape = input->Shape();

  // SwiGLU validation - FC3 not supported (match CUDA FasterTransformer)
  bool is_swiglu = (activation_type_ == MoEActivationType::SwiGLU);
  if (is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SwiGLU activation is not supported with fc3. Gate weights should be concatenated with FC1 weights.");
  }
  if (!is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FC3 gating is not yet implemented for non-SwiGLU activations on CPU.");
  }

  const int64_t act_multiplier = is_swiglu ? 2 : 1;
  const int64_t fc1_output_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
  const int64_t total_output_size = moe_params.num_rows * moe_params.hidden_size;

  const bool is_fp16 = input->DataType() == DataTypeImpl::GetType<MLFloat16>();

  //
  // step 1: run the gate program to get router indices and values
  //
  Tensor router_idx = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), input_shape);
  Tensor router_values = context.CreateGPUTensor(is_fp16 ? DataTypeImpl::GetType<MLFloat16>() : DataTypeImpl::GetType<float>(), input_shape);

  GateProgram gate{k_, is_fp16};
  gate
      .AddInputs({{input, ProgramTensorMetadataDependency::Type}})
      .AddOutput({&router_idx, ProgramTensorMetadataDependency::None})
      .AddOutput({&router_values, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize(moe_params.num_experts, moe_params.num_rows)
      .AddUniformVariables({static_cast<uint32_t>(moe_params.num_experts),
                            static_cast<uint32_t>(moe_params.num_rows)})
      .CacheHint(k_, moe_params.num_experts, is_fp16 ? "fp16" : "fp32");

  ORT_RETURN_IF_ERROR(context.RunProgram(gate));

  //
  // step 2: run the FC1 for the selected experts
  //

  //
  // step 3: apply activation
  //

  //
  // step 4: run the FC2 for the selected experts
  //

  //
  // step 5: merge selected experts
  //

  auto* output_tensor = context.Output(0, input_shape);
  int output_size = static_cast<int>(input_shape.Size());

  QMoEProgram program{input_shape, activation_type_};

  program
      .AddInputs({{input, ProgramTensorMetadataDependency::Type}})
      .AddInputs({{router_probs, ProgramTensorMetadataDependency::Type}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}})
      .CacheHint("hint?");

  return context.RunProgram(program);
}

namespace {
const std::vector<MLDataType>& QMoET1Constraint() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<uint8_t>()};
  return types;
}
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T1", QMoET1Constraint())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    QMoE);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
