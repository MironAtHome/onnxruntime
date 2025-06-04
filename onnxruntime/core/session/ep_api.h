// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace OrtExecutionProviderApi {
// implementation that returns the API struct
ORT_API(const OrtEpApi*, GetEpApi);

ORT_API_STATUS_IMPL(CreateEpDevice, _In_ OrtEpFactory* ep_factory,
                    _In_ const OrtHardwareDevice* hardware_device,
                    _In_opt_ const OrtKeyValuePairs* ep_metadata,
                    _In_opt_ const OrtKeyValuePairs* ep_options,
                    _Out_ OrtEpDevice** ep_device);

ORT_API(void, ReleaseEpDevice, _Frees_ptr_opt_ OrtEpDevice* device);

ORT_API_STATUS_IMPL(EpDevice_AddAllocatorInfo, _In_ OrtEpDevice* ep_device,
                    _In_ const OrtMemoryInfo* allocator_memory_info);

ORT_API(const OrtMemoryDevice*, OrtMemoryInfo_GetMemoryDevice, _In_ const OrtMemoryInfo* memory_info);
ORT_API_STATUS_IMPL(OrtValue_GetMemoryDevice, _In_ const OrtValue* value, _Out_ const OrtMemoryDevice** device);

ORT_API(bool, OrtMemoryDevice_AreEqual, _In_ const OrtMemoryDevice* a, _In_ const OrtMemoryDevice* b);
ORT_API(OrtMemoryInfoDeviceType, OrtMemoryDevice_GetDeviceType, _In_ const OrtMemoryDevice* memory_device);
ORT_API(OrtMemType, OrtMemoryDevice_GetMemoryType, _In_ const OrtMemoryDevice* memory_device);


ORT_API_STATUS_IMPL(CreateSyncStream, _In_ const OrtMemoryDevice* device, _In_ OrtSyncStreamImpl* impl,
                    _Outptr_ OrtSyncStream** stream);
ORT_API(OrtSyncStreamImpl*, SyncStream_GetStreamImpl, _In_ OrtSyncStream* stream);
ORT_API(const OrtMemoryDevice*, SyncStream_GetMemoryDevice, _In_ const OrtSyncStream* stream);
ORT_API(void, ReleaseSyncStream, _In_ OrtSyncStream* stream);

ORT_API_STATUS_IMPL(CreateSyncNotification, _In_ OrtSyncStream* stream, _In_ OrtSyncNotificationImpl* impl,
                    _Outptr_ OrtSyncNotification** notification);
ORT_API(void, ReleaseSyncNotification, _In_ OrtSyncNotification* notification);
}  // namespace OrtExecutionProviderApi
