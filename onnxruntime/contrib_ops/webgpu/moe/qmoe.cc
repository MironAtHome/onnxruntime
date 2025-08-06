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


Status QMoEProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
#if 0
  const auto& x_shape = shader.AddIndices("input_shape", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& indices = shader.AddInput("indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseIndicesToOffset);
  const auto& scales = shader.AddInput("scales", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody()
      << "var data_indices = input_shape_indices_t(0);\n"
      << "for (var i: u32 = 0; i < uniforms.gather_axis; i++) {\n"
      << "  let index = " << output.IndicesGet("output_indices", "i") << ";\n  "
      << x_shape.IndicesSet("data_indices", "i", "index") << ";\n};\n"
      << "var index_from_indices = " << indices.GetByIndices("indices_indices") << ";\n"
      << "if (index_from_indices < 0) { index_from_indices += " << x_shape_[gather_axis_] << ";}\n"
      << x_shape.IndicesSet("data_indices", "uniforms.gather_axis", "u32(index_from_indices)") << ";\n"
      << "for (var i = uniforms.gather_axis + 1; i < " << output_shape_.NumDimensions() << "; i++) {\n"
      << "  let index = " << output.IndicesGet("output_indices", "i + " + std::to_string(indices_rank_ - 1)) << ";\n  "
      << x_shape.IndicesSet("data_indices", "i", "index") << ";\n};\n"
      << "  let data_offset = " << x_shape.IndicesToOffset("data_indices") << ";\n";

  shader.MainFunctionBody()
      << "  let dequantized_data = (output_value_t(quantized_data) - output_value_t(zero_point)) * scale;\n  "
      << output.SetByOffset("global_idx", "dequantized_data") << ";\n";
#endif
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


  /*
    export_ids, router_weights = Gate(input, router_probs)
    for expert_id in export_ids:
      # nbitmm with expert_id[] passed in via input and index passed in via uniform?
      fc1_output = FC1(input, expert_id)
      fc1_output = Activation(fc1_output + fc1_bias)
      fc2_output = FC2(fc1_output, expert_id)
      output = output * fc2_output
  */

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
} // namespace

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
