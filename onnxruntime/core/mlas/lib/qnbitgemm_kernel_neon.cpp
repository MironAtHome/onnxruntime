/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qnbitgemm_kernel_neon.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON.

--*/

#include <arm_neon.h>

#include <cassert>

#include "qnbitgemm.h"
#include "qnbitgemm_kernel_neon.h"
#include "sqnbitgemm_q8_block.h"
#include "sqnbitgemm_tmac_kernel_neon_int8.h"

namespace sqnbitgemm_neon
{

namespace
{

//
// Quantized B data packing function implementation.
//

size_t
Q4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    constexpr size_t BlkBitWidth = 4;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    return PackedQuantBDataSize;
}

void
SQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    constexpr size_t BlkBitWidth = 4;

    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t Iterations = N * BlockCountK;  // one iteration per block

    const size_t SubBlkLen = (ComputeType == SQNBIT_CompInt8)
                                 ? ((BlkLen == 16) ? 16 : 32)
                                 : 16;

    const size_t SubBlkDataSize = SubBlkLen / 2;
    const size_t SubBlkBytePairCount = SubBlkLen / 4;

    //
    // For SubBlkLen == 16, pack 16 4-bit values (8 bytes) at a time like this:
    //
    // src: | v0 v1 | v2 v3 | v4 v5 | v6 v7 | v8 v9 | vA vB | vC vD | vE vF |
    //   =>
    // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
    //

    //
    // For SubBlkLen == 32, pack 32 4-bit values (16 bytes) at a time like this:
    //
    // src: | v0  v1  | v2  v3  | ... | v28 v29 | v30 v31 |
    //   =>
    // dst: | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
    //

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t n = tid / BlockCountK;
            const size_t k_blk = tid % BlockCountK;

            const size_t data_offset = n * BlockCountK * BlkDataSize + k_blk * BlkDataSize;
            const std::byte* QuantBData = QuantBDataBegin + data_offset;
            std::byte* PackedQuantBData = PackedQuantBDataBegin + data_offset;

            for (size_t kk = 0; kk < BlkLen; kk += SubBlkLen) {
                for (size_t byte_pair_idx = 0; byte_pair_idx < SubBlkBytePairCount; ++byte_pair_idx) {
                    const std::byte src0 = QuantBData[byte_pair_idx];
                    const std::byte src1 = QuantBData[byte_pair_idx + SubBlkDataSize / 2];

                    std::byte& dst0 = PackedQuantBData[2 * byte_pair_idx];
                    std::byte& dst1 = PackedQuantBData[2 * byte_pair_idx + 1];

                    dst0 = (src0 & std::byte{0x0F}) | ((src1 & std::byte{0x0F}) << 4);
                    dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
                }

                QuantBData += SubBlkDataSize;
                PackedQuantBData += SubBlkDataSize;
            }
        }
    );
}

//
// Workspace size calculation function implementation.
//

size_t
Q4BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            const size_t PerGemmWorkspaceSize = M * BlockCountK * Q8BlkSize(BlkLen);
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
}

size_t
Q4BitGemmPerGemmWorkspaceAlignment(
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            return Q8BlkAlignment();
        }
        default: {
            return 1;
        }
    }
}

size_t
Q2BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    constexpr size_t BlkBitWidth = 2;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    return PackedQuantBDataSize;
}

// Weight transform for data reuse
void SQ2BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool)
{
    constexpr size_t BITS = 2;                        // 2-bit quantization
    constexpr size_t ELEMENTS_PER_BYTE = 8 / BITS;    // 4 elements per byte
    constexpr size_t GROUP_SIZE = 32;                 // Optimal group size for cache locality
    
    // Validate block alignment requirements
    assert(BlkLen % 16 == 0 && "Block length must be multiple of 16 for SIMD alignment");
    
    // Configure SIMD width based on compute type
    const size_t SIMD_WIDTH = (ComputeType == SQNBIT_CompInt8) ? 32 : 16;

    // Calculate parallelization parameters
    const size_t block_count_k = (K + BlkLen - 1) / BlkLen;
    const size_t total_iterations = N * block_count_k;

    MlasTrySimpleParallel(ThreadPool, total_iterations,
        [&](ptrdiff_t task_id) {
            // Calculate matrix coordinates from task ID
            const size_t n = task_id / block_count_k;
            const size_t k_block = task_id % block_count_k;
            
            // Calculate data offsets for current block
            const size_t block_bytes = BlkLen * N / ELEMENTS_PER_BYTE;
            const size_t offset = n * block_bytes + k_block * (BlkLen / ELEMENTS_PER_BYTE);
            
            const std::byte* src = QuantBDataBegin + offset;
            std::byte* dst = PackedQuantBDataBegin + offset;

            // Temporary buffer for bit manipulation
            std::vector<uint8_t> unpacked_bits(BlkLen);

            // Process each SIMD-aligned sub-block
            for (size_t subblk = 0; subblk < BlkLen; subblk += SIMD_WIDTH) {
                // Phase 1: Bit unpacking - extract 2-bit elements
                for (size_t i = 0; i < SIMD_WIDTH; ++i) {
                    const size_t byte_idx = (subblk + i) / ELEMENTS_PER_BYTE;
                    const size_t bit_pos = 2 * ((subblk + i) % ELEMENTS_PER_BYTE);
                    unpacked_bits[i] = (static_cast<uint8_t>(src[byte_idx]) >> bit_pos) & 0x3;
                }

                // Phase 2: Group reorganization for SIMD efficiency
                for (size_t g = 0; g < SIMD_WIDTH; g += GROUP_SIZE) {
                    const size_t group_end = std::min(g + GROUP_SIZE, SIMD_WIDTH);
                    
                    // Realign elements within group
                    for (size_t i = g; i < group_end; ++i) {
                        // Calculate target position using matrix transformation:
                        // (original_index % group_size) * SIMD_WIDTH/group_size + original_index/group_size
                        const size_t new_pos = (i % GROUP_SIZE) * (SIMD_WIDTH / GROUP_SIZE) + (i / GROUP_SIZE);
                        
                        // Pack transformed values back into bytes
                        const size_t dst_byte = new_pos / ELEMENTS_PER_BYTE;
                        const size_t dst_shift = 2 * (new_pos % ELEMENTS_PER_BYTE);
                        dst[dst_byte] |= static_cast<std::byte>(unpacked_bits[i] << dst_shift);
                    }
                }

                // Advance pointers to next sub-block
                src += SIMD_WIDTH / ELEMENTS_PER_BYTE;
                dst += SIMD_WIDTH / ELEMENTS_PER_BYTE;
            }
        });
}

size_t
Q2BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            const size_t PerGemmWorkspaceSize = M * BlockCountK * Q8BlkSize(BlkLen);
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
}


size_t
Q2BitGemmPerGemmWorkspaceAlignment(
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            return Q8BlkAlignment();
        }
        default: {
            return 1;
        }
    }
}

void
QuantizeARowLUT_CompInt8 (
    size_t BlkLen,
    const float* A,
    std::byte* QuantA,
    size_t CountK,
    float* QuantAScale,
    float * QuantAZeroPoint
)
{
    if (CountK == 4096) {
        preprocessor_k4096(
            A,
            QuantAScale,
            QuantAZeroPoint,
            QuantA
        )
    }
    else if (CountK == 14336) {
        preprocessor_k14336(
            A,
            QuantAScale,
            QuantAZeroPoint,
            QuantA
        );
    }
    else {
        ORT_ENFORCE(false, "Unsupported shape: CountK=", CountK,
            ". Supported combinations: 4096, 14336");
    }
}



}  // namespace

}  // namespace sqnbitgemm_neon

//
// Kernel dispatch structure definition.
//

const MLAS_QNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchNeon = []() {
    MLAS_QNBIT_GEMM_DISPATCH d;

    d.Q4BitGemmPackQuantBDataSize = sqnbitgemm_neon::Q4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = sqnbitgemm_neon::SQ4BitGemmPackQuantBData;

    d.Q4BitGemmPerGemmWorkspaceSize = sqnbitgemm_neon::Q4BitGemmPerGemmWorkspaceSize;
    d.Q4BitGemmPerGemmWorkspaceAlignment = sqnbitgemm_neon::Q4BitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32 = sqnbitgemm_neon::SQ4BitGemmM1Kernel_CompFp32;
    d.SQ4BitBlkDequantBForSgemm_CompFp32 = sqnbitgemm_neon::SQ4BitBlkDequantBForSgemm_CompFp32;
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot()) {
        d.SQ4BitGemmKernel_CompInt8 = sqnbitgemm_neon::SQ4BitGemmKernel_CompInt8;
    }
    d.QuantizeARow_CompInt8 = sqnbitgemm_neon::QuantizeARow_CompInt8;

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    d.HQ4BitGemmPackQuantBData = sqnbitgemm_neon::HQ4BitGemmPackQuantBData_CompFp16;
    d.HQ4BitBlkDequantBForHgemm_CompFp16 = sqnbitgemm_neon::HQ4BitBlkDequantBForHgemm_CompFp16;
    d.HQ4BitGemmKernel_CompFp16 = sqnbitgemm_neon::HQ4BitGemmKernel_CompFp16;
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64

    d.Q2BitGemmPackQuantBDataSize = sqnbitgemm_neon::Q2BitGemmPackQuantBDataSize;
    d.SQ2BitGemmPackQuantBData = sqnbitgemm_neon::SQ2BitGemmPackQuantBData;

    d.Q2BitGemmPerGemmWorkspaceSize = sqnbitgemm_neon::Q2BitGemmPerGemmWorkspaceSize;

    d.SQ2BitGemmKernel_CompInt8 = sqnbitgemm_neon::SQ2BitGemmKernel_CompInt8;
    d.QuantizeARow_CompInt8 = sqnbitgemm_neon::QuantizeARow_CompInt8;
    return d;
}();
