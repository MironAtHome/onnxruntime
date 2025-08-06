// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/moe/moe_base.h"
#include "contrib_ops/webgpu/moe/moe.h"
#include "contrib_ops/cpu/quantization/moe_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;


Status MoEProgram::GenerateShaderCode(ShaderHelper& shader) const {
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

Status MoE::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input(0);                // (num_rows, hidden_size) or (batch_size, sequence_length, hidden_size)
  const Tensor* router_probs = context.Input(1);         // (num_rows, num_experts)
  const Tensor* fc1_experts_weights = context.Input(2);  // (num_experts, hidden_size, inter_size)
  const Tensor* fc1_experts_bias = context.Input(3);     // (num_experts, hidden_size, inter_size)
  const Tensor* fc2_experts_weights = context.Input(4);  // (num_experts, hidden_size, inter_size)
  const Tensor* fc2_experts_bias = context.Input(5);     // (num_experts, hidden_size, inter_size)
  const Tensor* fc3_experts_weights = context.Input(6);  // (num_experts, hidden_size, inter_size)
  const Tensor* fc3_experts_bias = context.Input(7);     // (num_experts, hidden_size, inter_size)
  const auto& input_shape = input->Shape();

  auto* output_tensor = context.Output(0, input_shape);
  int output_size = static_cast<int>(input_shape.Size());

  MoEProgram program{input_shape};

  program
      .AddInputs({{input, ProgramTensorMetadataDependency::Type}})
      .AddInputs({{router_probs, ProgramTensorMetadataDependency::Type}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}})
      .CacheHint("hint?");

  return context.RunProgram(program);
}

ONNX_OPERATOR_KERNEL_EX(
    MoE,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    MoE);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
