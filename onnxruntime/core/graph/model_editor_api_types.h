// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/ort_value.h"
#include "core/graph/abi_graph_types.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
struct ModelEditorValueInfo : public OrtValueInfo {
  ModelEditorValueInfo() : OrtValueInfo(OrtGraphIrApi::kModelEditorApi) {}

  // Defines ToExternal() and ToInternal() functions to convert between OrtValueInfo and ModelEditorValueInfo.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtValueInfo, ModelEditorValueInfo, OrtGraphIrApi::kModelEditorApi)

  const std::string& Name() const override { return name; }
  const OrtTypeInfo* TypeInfo() const override { return type_info.get(); }
  Status GetProducerInfo(ProducerInfo& /*producer_info*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the producer for OrtValueInfo");
  }
  Status GetConsumerInfos(std::vector<ConsumerInfo>& /*consumer_infos*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the users for a OrtValueInfo");
  }
  Status GetNumConsumers(size_t& /*num_consumers*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the number of users for a OrtValueInfo");
  }

  std::string name;
  std::unique_ptr<OrtTypeInfo> type_info;
};

struct ModelEditorNode : public OrtNode {
  ModelEditorNode() : OrtNode(OrtGraphIrApi::kModelEditorApi) {}

  // Defines ToExternal() and ToInternal() functions to convert between OrtNode and ModelEditorNode.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtNode, ModelEditorNode, OrtGraphIrApi::kModelEditorApi)

  const std::string& Name() const override { return node_name; }
  const std::string& OpType() const override { return operator_name; }
  const std::string& Domain() const override { return domain_name; }
  size_t GetNumInputs() const override { return input_names.size(); }
  size_t GetNumOutputs() const override { return output_names.size(); }

  Status GetInputs(InlinedVector<const OrtValueInfo*>& /*inputs*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting input OrtValueInfos for OrtNode");
  }
  Status GetOutputs(InlinedVector<const OrtValueInfo*>& /*outputs*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting output OrtValueInfos for OrtNode");
  }

  std::string operator_name;
  std::string domain_name;
  std::string node_name;

  // OrtOpAttr is 1:1 with ONNX_NAMESPACE::AttributeProto currently.
  // https://github.com/microsoft/onnxruntime/blob/bd5a759d0cdbed6e7f611c990d4eb5457a9ecf60/onnxruntime/core/session/standalone_op_invoker.cc#L318
  onnxruntime::InlinedVector<ONNX_NAMESPACE::AttributeProto> attributes;
  onnxruntime::InlinedVector<std::string> input_names;
  onnxruntime::InlinedVector<std::string> output_names;

  // FUTURE if we need control flow nodes
  // std::unordered_map<std::string, OrtGraph> subgraphs;
};

struct ModelEditorGraph : public OrtGraph {
  ModelEditorGraph() : OrtGraph(OrtGraphIrApi::kModelEditorApi) {}

  // Defines ToExternal() and ToInternal() functions to convert between OrtGraph and ModelEditorGraph.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtGraph, ModelEditorGraph, OrtGraphIrApi::kModelEditorApi)
  const std::string& Name() const override { return name; }
  size_t NumberOfNodes() const override { return nodes.size(); }
  std::vector<const OrtNode*> GetNodes(int /*order*/) const override {
    std::vector<const OrtNode*> result;
    result.reserve(nodes.size());
    for (const auto& n : nodes) {
      result.push_back(n.get());
    }
    return result;
  }

  onnxruntime::InlinedVector<std::unique_ptr<onnxruntime::ModelEditorValueInfo>> inputs;
  onnxruntime::InlinedVector<std::unique_ptr<onnxruntime::ModelEditorValueInfo>> outputs;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> initializers;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> external_initializers;
  std::vector<std::unique_ptr<onnxruntime::ModelEditorNode>> nodes;
  std::string name = "ModelEditorGraph";
};

}  // namespace onnxruntime

struct OrtModel {
  std::unique_ptr<OrtGraph> graph;
  std::unordered_map<std::string, int> domain_to_version;
};
