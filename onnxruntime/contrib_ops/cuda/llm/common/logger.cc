/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/common/cudaUtils.h"
#include <cuda_runtime.h>

#include "core/common/common.h"


namespace ort_llm::common
{

Logger::Logger()
{
    char* isFirstRankOnlyChar = std::getenv("TLLM_LOG_FIRST_RANK_ONLY");
    bool isFirstRankOnly = (isFirstRankOnlyChar != nullptr && std::string(isFirstRankOnlyChar) == "ON");

    auto const* levelName = std::getenv("TLLM_LOG_LEVEL");
    if (levelName != nullptr)
    {
        auto level = [levelName = std::string(levelName)]()
        {
            if (levelName == "TRACE")
                return TRACE;
            if (levelName == "VERBOSE" || levelName == "DEBUG")
                return DEBUG;
            if (levelName == "INFO")
                return INFO;
            if (levelName == "WARNING")
                return WARNING;
            if (levelName == "ERROR")
                return ERROR;
            ORT_THROW("Invalid log level:", levelName);
        }();
        // If TLLM_LOG_FIRST_RANK_ONLY=ON, set LOG LEVEL of other device to ERROR
        if (isFirstRankOnly)
        {
            auto const deviceId = getDevice();
            if (deviceId != 1)
            {
                level = ERROR;
            }
        }
        setLevel(level);
    }
}

Logger* Logger::getLogger()
{
    thread_local Logger instance;
    return &instance;
}
} // namespace ort_llm::common
