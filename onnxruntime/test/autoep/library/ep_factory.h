// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "ep_data_transfer.h"
#include "utils.h"

class ExampleEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  ExampleEpFactory(const char* ep_name, ApiPtrs apis);

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr);

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr);

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices);

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              const OrtHardwareDevice* const* /*devices*/,
                                              const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger,
                                              OrtEp** ep);

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep);

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

  // get the shared OrtDataTransferImpl instance.
  const OrtDataTransferImpl* GetDataTransferImpl() noexcept {
    return data_transfer_impl_.get();
  }

  const std::string ep_name_;            // EP name
  const std::string vendor_{"Contoso"};  // EP vendor name

 private:
  // CPU allocator so we can control the arena behavior. optional as ORT always provides a CPU allocator if needed.
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
  MemoryInfoUniquePtr cpu_memory_info_;

  // for example purposes. if the EP used GPU, and pinned/shared memory was required for data transfer, these are the
  // OrtMemoryInfo instance required for that.
  MemoryInfoUniquePtr default_gpu_memory_info_;
  MemoryInfoUniquePtr pinned_gpu_memory_info_;

  std::unique_ptr<ExampleDataTransfer> data_transfer_impl_;  // data transfer implementation for this factory
};
