#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define RETURN_IF_ERROR(fn)   \
  do {                        \
    OrtStatus* status = (fn); \
    if (status != nullptr) {  \
      return status;          \
    }                         \
  } while (0)

#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

struct ExampleEp;

/// <summary>
/// Example implementation of ONNX Mul. Does not handle many things like broadcasting.
/// </summary>
struct MulKernel {
  MulKernel(const OrtApi& ort_api, const OrtLogger& logger) : ort_api(ort_api), logger(logger) {}
  ~MulKernel() { ReleaseResources(); }

  OrtStatus* Compute(OrtKernelContext* kernel_context) {
    ReleaseResources();
    RETURN_IF_ERROR(ort_api.Logger_LogMessage(&logger,
                                              OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                              "MulKernel::Compute", ORT_FILE, __LINE__, __FUNCTION__));
    size_t num_inputs = 0;
    RETURN_IF_ERROR(ort_api.KernelContext_GetInputCount(kernel_context, &num_inputs));
    RETURN_IF(num_inputs != 2, ort_api, "Expected 2 inputs for MulKernel");

    size_t num_outputs = 0;
    RETURN_IF_ERROR(ort_api.KernelContext_GetOutputCount(kernel_context, &num_outputs));
    RETURN_IF(num_outputs != 1, ort_api, "Expected 1 output for MulKernel");

    const OrtValue* input0 = nullptr;
    const OrtValue* input1 = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetInput(kernel_context, 0, &input0));
    RETURN_IF_ERROR(ort_api.KernelContext_GetInput(kernel_context, 1, &input1));

    type_shape0 = nullptr;
    type_shape1 = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(input0, &type_shape0));
    RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(input1, &type_shape1));

    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(ort_api.GetTensorElementType(type_shape0, &elem_type));
    RETURN_IF(elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ort_api, "Expected float32 inputs");

    size_t num_dims0 = 0;
    size_t num_dims1 = 0;
    RETURN_IF_ERROR(ort_api.GetDimensionsCount(type_shape0, &num_dims0));
    RETURN_IF_ERROR(ort_api.GetDimensionsCount(type_shape1, &num_dims1));
    RETURN_IF((num_dims0 == 0) || (num_dims1 == 0), ort_api, "Input has 0 dimensions");
    RETURN_IF(num_dims0 != num_dims1, ort_api, "Expected same dimensions for both inputs");  // No broadcasting

    std::vector<int64_t> dims0(num_dims0, 0);
    std::vector<int64_t> dims1(num_dims1, 0);
    RETURN_IF_ERROR(ort_api.GetDimensions(type_shape0, dims0.data(), dims0.size()));
    RETURN_IF_ERROR(ort_api.GetDimensions(type_shape1, dims1.data(), dims1.size()));
    RETURN_IF(dims0 != dims1, ort_api, "Expected same dimensions for both inputs");  // No broadcasting.

    const float* input_data0 = nullptr;
    const float* input_data1 = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(const_cast<OrtValue*>(input0), (void**)&input_data0));  // No const-correct API?
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(const_cast<OrtValue*>(input1), (void**)&input_data1));

    OrtValue* output = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetOutput(kernel_context, 0, dims0.data(), dims0.size(), &output));

    float* output_data = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(output, reinterpret_cast<void**>(&output_data)));

    int64_t num_elems = 1;
    for (int64_t dim : dims0) {
      RETURN_IF(dim < 0, ort_api, "Invalid dimension: negative value detected");
      num_elems *= dim;
    }

    for (size_t i = 0; i < static_cast<size_t>(num_elems); ++i) {
      output_data[i] = input_data0[i] * input_data1[i];
    }

    return nullptr;
  }

  void ReleaseResources() noexcept {
    // Note: resource cleanup would be simplified with the C++ ORT API, but sticking to C API for now.
    if (type_shape0 != nullptr) {
      ort_api.ReleaseTensorTypeAndShapeInfo(type_shape0);
      type_shape0 = nullptr;
    }

    if (type_shape1 != nullptr) {
      ort_api.ReleaseTensorTypeAndShapeInfo(type_shape1);
      type_shape1 = nullptr;
    }
  }

  const OrtApi& ort_api;
  const OrtLogger& logger;
  OrtTensorTypeAndShapeInfo* type_shape0 = nullptr;
  OrtTensorTypeAndShapeInfo* type_shape1 = nullptr;
};

/// <summary>
/// Example OrtNodeComputeInfo that represents the computation function for a compiled OrtGraph.
/// </summary>
struct ExampleNodeComputeInfo : OrtNodeComputeInfo {
  explicit ExampleNodeComputeInfo(ExampleEp& ep);

  static OrtStatus* ORT_API_CALL CreateComputeStateImpl(OrtNodeComputeInfo* this_ptr,
                                                        OrtNodeComputeContext* compute_context,
                                                        void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             const OrtApi* api, OrtKernelContext* kernel_context);
  static void ORT_API_CALL DestroyComputeStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  ExampleEp& ep;
};

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

static OrtStatus* IsFloatTensor(const OrtApi& ort_api, const OrtValueInfo* value_info, bool& result) {
  result = false;

  const OrtTypeInfo* type_info = nullptr;
  RETURN_IF_ERROR(ort_api.GetValueInfoTypeInfo(value_info, &type_info));

  ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
  RETURN_IF_ERROR(ort_api.GetOnnxTypeFromTypeInfo(type_info, &onnx_type));
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return nullptr;
  }

  const OrtTensorTypeAndShapeInfo* type_shape = nullptr;
  RETURN_IF_ERROR(ort_api.CastTypeInfoToTensorInfo(type_info, &type_shape));

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  RETURN_IF_ERROR(ort_api.GetTensorElementType(type_shape, &elem_type));
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return nullptr;
  }

  result = true;
  return nullptr;
}

/// <summary>
/// Example EP that can compile a single Mul operator.
/// </summary>
struct ExampleEp : OrtEp, ApiPtrs {
  ExampleEp(ApiPtrs apis, const std::string& name, const OrtHardwareDevice& device,
            const OrtSessionOptions& session_options, const OrtLogger& logger)
      : ApiPtrs(apis), name_{name}, hardware_device_{device}, session_options_{session_options}, logger_{logger} {
    // Initialize the execution provider.
    auto status = ort_api.Logger_LogMessage(&logger_,
                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                            ("ExampleEp has been created with name " + name_).c_str(),
                                            ORT_FILE, __LINE__, __FUNCTION__);
    // ignore status for now
    (void)status;

    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
    GetName = GetNameImpl;
    GetCapability = GetCapabilityImpl;
    Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  }

  ~ExampleEp() {
    // Clean up the execution provider
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) {
    const auto* ep = static_cast<const ExampleEp*>(this_ptr);
    return ep->name_.c_str();
  }

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) {
    ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);

    size_t num_nodes = ep->ep_api.Graph_GetNumNodes(graph);
    if (num_nodes == 0) {
      return nullptr;  // No nodes to process
    }

    std::vector<const OrtNode*> nodes(num_nodes, nullptr);
    RETURN_IF_ERROR(ep->ep_api.Graph_GetNodes(graph, /*order*/ 0, nodes.data(), nodes.size()));

    std::vector<const OrtNode*> supported_nodes;

    for (const OrtNode* node : nodes) {
      const char* op_type = ep->ep_api.Node_GetOperatorType(node);

      if (std::strncmp(op_type, "Mul", 4) == 0) {
        // Check that Mul has inputs/output of type float
        size_t num_inputs = ep->ep_api.Node_GetNumInputs(node);
        size_t num_outputs = ep->ep_api.Node_GetNumOutputs(node);
        RETURN_IF(num_inputs != 2 || num_outputs != 1, ep->ort_api, "Mul should have 2 inputs and 1 output");

        std::vector<const OrtValueInfo*> inputs(num_inputs, nullptr);
        std::vector<const OrtValueInfo*> outputs(num_outputs, nullptr);
        RETURN_IF_ERROR(ep->ep_api.Node_GetInputs(node, inputs.data(), inputs.size()));
        RETURN_IF_ERROR(ep->ep_api.Node_GetOutputs(node, outputs.data(), outputs.size()));

        std::array<bool, 3> is_float_tensor = {false, false, false};
        RETURN_IF_ERROR(IsFloatTensor(ep->ort_api, inputs[0], is_float_tensor[0]));
        RETURN_IF_ERROR(IsFloatTensor(ep->ort_api, inputs[1], is_float_tensor[1]));
        RETURN_IF_ERROR(IsFloatTensor(ep->ort_api, outputs[0], is_float_tensor[2]));
        if (!is_float_tensor[0] || !is_float_tensor[1] || !is_float_tensor[2]) {
          continue;  // Input or output is not of type float
        }

        supported_nodes.push_back(node);  // Only support a single Mul for now.
        break;
      }
    }
    RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddSupportedNodes(graph_support_info, supported_nodes.data(),
                                                                    supported_nodes.size(), &ep->hardware_device_));
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs, size_t num_graphs,
                                             OrtNodeComputeInfo** node_compute_infos) {
    ExampleEp* ep = static_cast<ExampleEp*>(this_ptr);

    if (num_graphs != 1) {
      return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single graph");
    }

    size_t num_nodes = ep->ep_api.Graph_GetNumNodes(graphs[0]);

    std::vector<const OrtNode*> nodes(num_nodes, nullptr);
    RETURN_IF_ERROR(ep->ep_api.Graph_GetNodes(graphs[0], /*order*/ 0, nodes.data(), nodes.size()));

    if (num_nodes != 1) {
      return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single Mul node");
    }

    const char* node_op_type = ep->ep_api.Node_GetOperatorType(nodes[0]);
    if (std::strncmp(node_op_type, "Mul", 4) != 0) {
      return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Expected to compile a single Mul node");
    }

    // Now we know we're compiling a single Mul node.
    // Associate the graph name (aka fused node name) with our MulKernel.
    const char* fused_node_name = ep->ep_api.Graph_GetName(graphs[0]);
    ep->kernels[fused_node_name] = std::make_unique<MulKernel>(ep->ort_api, ep->logger_);

    // Update the OrtNodeComputeInfo associated with the graph.
    auto node_compute_info = std::make_unique<ExampleNodeComputeInfo>(*ep);
    node_compute_infos[0] = node_compute_info.release();

    return nullptr;
  }

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) {
    (void)this_ptr;
    for (size_t i = 0; i < num_node_compute_infos; i++) {
      delete node_compute_infos[i];
    }
  }

  std::string name_;
  const OrtHardwareDevice& hardware_device_;
  const OrtSessionOptions& session_options_;
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<MulKernel>> kernels;
};

//
// Implementation of ExampleNodeComuteInfo
//
ExampleNodeComputeInfo::ExampleNodeComputeInfo(ExampleEp& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateComputeState = CreateComputeStateImpl;
  Compute = ComputeImpl;
  DestroyComputeState = DestroyComputeStateImpl;
}

OrtStatus* ExampleNodeComputeInfo::CreateComputeStateImpl(OrtNodeComputeInfo* this_ptr,
                                                          OrtNodeComputeContext* compute_context,
                                                          void** compute_state) {
  auto* node_compute_info = static_cast<ExampleNodeComputeInfo*>(this_ptr);
  ExampleEp& ep = node_compute_info->ep;

  std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  auto kernel_it = ep.kernels.find(fused_node_name);
  if (kernel_it == ep.kernels.end()) {
    std::string message = "Unable to get kernel for fused node with name " + fused_node_name;
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  MulKernel& kernel = *kernel_it->second;
  *compute_state = &kernel;
  return nullptr;
}

OrtStatus* ExampleNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                               const OrtApi* /*ort_api*/, OrtKernelContext* kernel_context) {
  (void)this_ptr;
  MulKernel& kernel = *reinterpret_cast<MulKernel*>(compute_state);
  return kernel.Compute(kernel_context);
}

void ExampleNodeComputeInfo::DestroyComputeStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  MulKernel& kernel = *reinterpret_cast<MulKernel*>(compute_state);
  kernel.ReleaseResources();
}

/// <summary>
/// Example EP factory that can create an OrtEp and return information about the supported hardware devices.
/// </summary>
struct ExampleEpFactory : OrtEpFactory, ApiPtrs {
  ExampleEpFactory(const char* ep_name, ApiPtrs apis) : ApiPtrs(apis), ep_name_{ep_name} {
    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
    return factory->ep_name_.c_str();
  }

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
    return factory->vendor_.c_str();
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<ExampleEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      // C API
      const OrtHardwareDevice& device = *devices[i];
      if (factory->ort_api.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
        // these can be returned as nullptr if you have nothing to add.
        OrtKeyValuePairs* ep_metadata = nullptr;
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_metadata);
        factory->ort_api.CreateKeyValuePairs(&ep_options);

        // random example using made up values
        factory->ort_api.AddKeyValuePair(ep_metadata, "version", "0.1");
        factory->ort_api.AddKeyValuePair(ep_options, "run_really_fast", "true");

        // OrtEpDevice copies ep_metadata and ep_options.
        auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                   &ep_devices[num_ep_devices++]);

        factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
        factory->ort_api.ReleaseKeyValuePairs(ep_options);

        if (status != nullptr) {
          return status;
        }
      }

      // C++ API equivalent. Throws on error.
      //{
      //  Ort::ConstHardwareDevice device(devices[i]);
      //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      //    Ort::KeyValuePairs ep_metadata;
      //    Ort::KeyValuePairs ep_options;
      //    ep_metadata.Add("version", "0.1");
      //    ep_options.Add("run_really_fast", "true");
      //    Ort::EpDevice ep_device{*this_ptr, device, ep_metadata.GetConst(), ep_options.GetConst()};
      //    ep_devices[num_ep_devices++] = ep_device.release();
      //  }
      //}
    }

    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                              _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              _In_ size_t num_devices,
                                              _In_ const OrtSessionOptions* session_options,
                                              _In_ const OrtLogger* logger,
                                              _Out_ OrtEp** ep) {
    auto* factory = static_cast<ExampleEpFactory*>(this_ptr);
    *ep = nullptr;

    if (num_devices != 1) {
      // we only registered for CPU and only expected to be selected for one CPU
      // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
      // the EP has been selected for.
      return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                           "Example EP only supports selection for one device.");
    }

    // Create the execution provider
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       "Creating Example EP", ORT_FILE, __LINE__, __FUNCTION__));

    // use properties from the device and ep_metadata if needed
    // const OrtHardwareDevice* device = devices[0];
    // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

    auto dummy_ep = std::make_unique<ExampleEp>(*factory, factory->ep_name_, *devices[0], *session_options, *logger);

    *ep = dummy_ep.release();
    return nullptr;
  }

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) {
    ExampleEp* dummy_ep = static_cast<ExampleEp*>(ep);
    delete dummy_ep;
  }

  const std::string ep_name_;            // EP name
  const std::string vendor_{"Contoso"};  // EP vendor name
};

// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ort_ep_api = ort_api->GetEpApi();

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<ExampleEpFactory>(registration_name,
                                                                             ApiPtrs{*ort_api, *ort_ep_api});

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete factory;
  return nullptr;
}

}  // extern "C"
