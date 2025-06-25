/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocsparse_csrgemm_nnz_calc.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "control.h"
#include "csrgemm_device.h"
#include "rocsparse_csrgemm.hpp"
#include "utility.h"

#include "rocsparse_primitives.h"

#include <vector>

namespace rocsparse
{
    template <uint32_t HASHSIZE, typename J>
    constexpr uint32_t csrgemm_nnz_block_per_row_shared_memory_size()
    {
        return (sizeof(J) * HASHSIZE);
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_nnz_calc(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             J                         m,
                                             J                         n,
                                             J                         k,
                                             const rocsparse_mat_descr descr_A,
                                             I                         nnz_A,
                                             const I*                  csr_row_ptr_A,
                                             const J*                  csr_col_ind_A,
                                             const rocsparse_mat_descr descr_B,
                                             I                         nnz_B,
                                             const I*                  csr_row_ptr_B,
                                             const J*                  csr_col_ind_B,
                                             const rocsparse_mat_descr descr_D,
                                             I                         nnz_D,
                                             const I*                  csr_row_ptr_D,
                                             const J*                  csr_col_ind_D,
                                             const rocsparse_mat_descr descr_C,
                                             I*                        csr_row_ptr_C,
                                             I*                        nnz_C,
                                             const rocsparse_mat_info  info_C,
                                             void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

    // Index base
    rocsparse_index_base base_A
        = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
    rocsparse_index_base base_B
        = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
    rocsparse_index_base base_D = info_C->csrgemm_info->add
                                      ? ((descr_D) ? descr_D->base : rocsparse_index_base_zero)
                                      : rocsparse_index_base_zero;

    std::vector<I> hcsr_row_ptr_C(m + 1, 0);
    std::vector<J> h_group_size   = {38, 11, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<J> h_group_offset = {0, 38, 49, 50, 50, 50, 50, 50, 50, 50, 50};
    std::vector<J> h_perm = {0,  1,  2,  3,  7,  8,  10, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23,
                             24, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44,
                             45, 46, 47, 48, 4,  5,  6,  9,  11, 17, 18, 25, 29, 43, 49, 27};

    J* d_group_offset = nullptr;
    J* d_perm         = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc((void**)&d_group_offset, sizeof(J) * 11));
    RETURN_IF_HIP_ERROR(hipMalloc((void**)&d_perm, sizeof(J) * m));

    RETURN_IF_HIP_ERROR(
        hipMemcpy(d_group_offset, h_group_offset.data(), sizeof(J) * 11, hipMemcpyHostToDevice));
    RETURN_IF_HIP_ERROR(hipMemcpy(d_perm, h_perm.data(), sizeof(J) * m, hipMemcpyHostToDevice));

    // std::cout << "h_group_size" << std::endl;
    // for(size_t i = 0; i < CSRGEMM_MAXGROUPS; i++)
    // {
    //     std::cout << h_group_size[i] << " ";
    // }
    // std::cout << "" << std::endl;

    // RETURN_IF_HIP_ERROR(hipMemcpy(h_group_offset.data(),
    //                               d_group_offset,
    //                               sizeof(J) * CSRGEMM_MAXGROUPS,
    //                               hipMemcpyDeviceToHost));
    // std::cout << "h_group_offset" << std::endl;
    // for(size_t i = 0; i < h_group_offset.size(); i++)
    // {
    //     std::cout << h_group_offset[i] << " ";
    // }
    // std::cout << "" << std::endl;

    // RETURN_IF_HIP_ERROR(hipMemcpy(h_perm.data(), d_perm, sizeof(J) * m, hipMemcpyDeviceToHost));
    // std::cout << "h_perm" << std::endl;
    // for(size_t i = 0; i < h_perm.size(); i++)
    // {
    //     std::cout << h_perm[i] << " ";
    // }
    // std::cout << "" << std::endl;

    // Group 2: 65 - 512 intermediate products
    //if(h_group_size[2] > 0)
    //{
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 512
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                              CSRGEMM_SUB,
                                              CSRGEMM_HASHSIZE,
                                              CSRGEMM_NNZ_HASH>),
        dim3(h_group_size[2]),
        dim3(CSRGEMM_DIM),
        (csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()),
        stream,
        &d_group_offset[2],
        d_perm,
        csr_row_ptr_A,
        csr_col_ind_A,
        csr_row_ptr_B,
        csr_col_ind_B,
        csr_row_ptr_D,
        csr_col_ind_D,
        csr_row_ptr_C,
        base_A,
        base_B,
        base_D,
        info_C->csrgemm_info->mul,
        info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    //}

    RETURN_IF_HIP_ERROR(hipMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(I) * (m + 1), hipMemcpyDeviceToHost));
    std::cout << "After group 2 hcsr_row_ptr_C" << std::endl;
    for(size_t i = 0; i < hcsr_row_ptr_C.size(); i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    RETURN_IF_HIP_ERROR(hipFree(d_group_offset));
    RETURN_IF_HIP_ERROR(hipFree(d_perm));

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J)                                                                          \
    template rocsparse_status rocsparse::csrgemm_nnz_calc(rocsparse_handle          handle,        \
                                                          rocsparse_operation       trans_A,       \
                                                          rocsparse_operation       trans_B,       \
                                                          J                         m,             \
                                                          J                         n,             \
                                                          J                         k,             \
                                                          const rocsparse_mat_descr descr_A,       \
                                                          I                         nnz_A,         \
                                                          const I*                  csr_row_ptr_A, \
                                                          const J*                  csr_col_ind_A, \
                                                          const rocsparse_mat_descr descr_B,       \
                                                          I                         nnz_B,         \
                                                          const I*                  csr_row_ptr_B, \
                                                          const J*                  csr_col_ind_B, \
                                                          const rocsparse_mat_descr descr_D,       \
                                                          I                         nnz_D,         \
                                                          const I*                  csr_row_ptr_D, \
                                                          const J*                  csr_col_ind_D, \
                                                          const rocsparse_mat_descr descr_C,       \
                                                          I*                        csr_row_ptr_C, \
                                                          I*                        nnz_C,         \
                                                          const rocsparse_mat_info  info_C,        \
                                                          void*                     temp_buffer)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);

#undef INSTANTIATE
