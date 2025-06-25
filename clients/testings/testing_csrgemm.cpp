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

#include "testing.hpp"

//
//
//
template <typename T>
void testing_csrgemm_bad_arg(const Arguments& arg)
{
    static const size_t              safe_size = 1;
    T                                h_alpha   = 0.6;
    T                                h_beta    = 0.2;
    host_dense_vector<rocsparse_int> hcsr_row_ptr_C(safe_size + 1);
    hcsr_row_ptr_C[0] = 0;
    hcsr_row_ptr_C[1] = 1;
    device_dense_vector<rocsparse_int> dcsr_row_ptr_C(hcsr_row_ptr_C);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;
    // Create matrix descriptors
    rocsparse_local_mat_descr local_descr_A;
    rocsparse_local_mat_descr local_descr_B;
    rocsparse_local_mat_descr local_descr_C;
    rocsparse_local_mat_descr local_descr_D;

    // Create info desciptor
    rocsparse_local_mat_info local_info_C;

    rocsparse_handle    handle      = local_handle;
    rocsparse_operation trans_A     = rocsparse_operation_none;
    rocsparse_operation trans_B     = rocsparse_operation_none;
    rocsparse_int       m           = safe_size;
    rocsparse_int       n           = safe_size;
    rocsparse_int       k           = safe_size;
    rocsparse_mat_info  info_C      = local_info_C;
    size_t*             buffer_size = (size_t*)0x4;
    void*               temp_buffer = (void*)0x4;

    // C matrix
    const rocsparse_mat_descr descr_C       = local_descr_C;
    T*                        csr_val_C     = (T*)0x4;
    rocsparse_int*            csr_row_ptr_C = (rocsparse_int*)dcsr_row_ptr_C;
    rocsparse_int*            csr_col_ind_C = (rocsparse_int*)0x4;
    rocsparse_int*            nnz_C         = (rocsparse_int*)0x4;

#define PARAMS_BUFFER_SIZE                                                                  \
    handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, \
        descr_B, nnz_B, csr_row_ptr_B, csr_col_ind_B, beta, descr_D, nnz_D, csr_row_ptr_D,  \
        csr_col_ind_D, info_C, buffer_size

#define PARAMS_NNZ                                                                            \
    handle, trans_A, trans_B, m, n, k, descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, descr_B, \
        nnz_B, csr_row_ptr_B, csr_col_ind_B, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D,    \
        descr_C, csr_row_ptr_C, nnz_C, info_C, temp_buffer

#define PARAMS                                                                                 \
    handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_val_A, csr_row_ptr_A,        \
        csr_col_ind_A, descr_B, nnz_B, csr_val_B, csr_row_ptr_B, csr_col_ind_B, beta, descr_D, \
        nnz_D, csr_val_D, csr_row_ptr_D, csr_col_ind_D, descr_C, csr_val_C, csr_row_ptr_C,     \
        csr_col_ind_C, info_C, temp_buffer

    // 4 Scenarios need to be tested:

    // Scenario 1: alpha == nullptr && beta == nullptr
    // Scenario 2: alpha != nullptr && beta == nullptr
    // Scenario 3: alpha == nullptr && beta != nullptr
    // Scenario 4: alpha != nullptr && beta != nullptr

    // ###############################################
    // Scenario 1: alpha == nullptr && beta == nullptr
    // ###############################################
    {
        // In this scenario matrices A == B == D == nullptr
        static constexpr int nargs_to_exclude_buffer_size = 14;
        static constexpr int nargs_to_exclude_nnz         = 16;
        static constexpr int nargs_to_exclude             = 22;

        const int args_to_exclude_buffer_size[nargs_to_exclude_buffer_size]
            = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        const int args_to_exclude_nnz[nargs_to_exclude_nnz]
            = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22};
        const int args_to_exclude[nargs_to_exclude]
            = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28};

        const T* alpha = (const T*)nullptr;
        const T* beta  = (const T*)nullptr;

        // A matrix
        const rocsparse_mat_descr descr_A       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_A         = 0;
        const T*                  csr_val_A     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)nullptr;

        // B matrix
        const rocsparse_mat_descr descr_B       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_B         = 0;
        const T*                  csr_val_B     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)nullptr;

        // D matrix
        const rocsparse_mat_descr descr_D       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_D         = 0;
        const T*                  csr_val_D     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_D = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_D = (const rocsparse_int*)nullptr;
        select_bad_arg_analysis(rocsparse_csrgemm_buffer_size<T>,
                                nargs_to_exclude_buffer_size,
                                args_to_exclude_buffer_size,
                                PARAMS_BUFFER_SIZE);

        select_bad_arg_analysis(
            rocsparse_csrgemm_nnz, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);
        select_bad_arg_analysis(rocsparse_csrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 2: alpha != nullptr && beta == nullptr
    // ###############################################
    {
        // In this scenario matrices A != B != nullptr and D == nullptr
        static constexpr int nargs_to_exclude_buffer_size = 6;
        static constexpr int nargs_to_exclude_nnz         = 4;
        static constexpr int nargs_to_exclude             = 6;

        const int args_to_exclude_buffer_size[nargs_to_exclude_buffer_size]
            = {6, 15, 16, 17, 18, 19};
        const int args_to_exclude_nnz[nargs_to_exclude_nnz] = {14, 15, 16, 17};
        const int args_to_exclude[nargs_to_exclude]         = {17, 18, 19, 20, 21, 22};

        const T* alpha = &h_alpha;
        const T* beta  = (const T*)nullptr;

        // A matrix
        const rocsparse_mat_descr descr_A       = local_descr_A;
        rocsparse_int             nnz_A         = safe_size;
        const T*                  csr_val_A     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)0x4;

        // B matrix
        const rocsparse_mat_descr descr_B       = local_descr_B;
        rocsparse_int             nnz_B         = safe_size;
        const T*                  csr_val_B     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)0x4;

        // D matrix
        const rocsparse_mat_descr descr_D       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_D         = 0;
        const T*                  csr_val_D     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_D = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_D = (const rocsparse_int*)nullptr;

        select_bad_arg_analysis(rocsparse_csrgemm_buffer_size<T>,
                                nargs_to_exclude_buffer_size,
                                args_to_exclude_buffer_size,
                                PARAMS_BUFFER_SIZE);

        select_bad_arg_analysis(
            rocsparse_csrgemm_nnz, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);

        select_bad_arg_analysis(rocsparse_csrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }
    // ###############################################
    // Scenario 3: alpha == nullptr && beta != nullptr
    // ###############################################
    {
        // In this scenario matrices A == B == nullptr and D != nullptr
        static constexpr int nargs_to_exclude_buffer_size = 11;
        static constexpr int nargs_to_exclude_nnz         = 9;
        static constexpr int nargs_to_exclude             = 12;

        const int args_to_exclude_buffer_size[nargs_to_exclude_buffer_size]
            = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        const int args_to_exclude_nnz[nargs_to_exclude_nnz] = {6, 7, 8, 9, 10, 11, 12, 13, 22};
        const int args_to_exclude[nargs_to_exclude] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 28};

        const T* alpha = (const T*)nullptr;
        const T* beta  = &h_beta;

        temp_buffer = (void*)nullptr;

        // A matrix
        const rocsparse_mat_descr descr_A       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_A         = 0;
        const T*                  csr_val_A     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)nullptr;

        // B matrix
        const rocsparse_mat_descr descr_B       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnz_B         = 0;
        const T*                  csr_val_B     = (const T*)nullptr;
        const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)nullptr;
        const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)nullptr;

        // D matrix
        const rocsparse_mat_descr descr_D       = local_descr_D;
        rocsparse_int             nnz_D         = safe_size;
        const T*                  csr_val_D     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_D = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_D = (const rocsparse_int*)0x4;
        select_bad_arg_analysis(rocsparse_csrgemm_buffer_size<T>,
                                nargs_to_exclude_buffer_size,
                                args_to_exclude_buffer_size,
                                PARAMS_BUFFER_SIZE);

        select_bad_arg_analysis(
            rocsparse_csrgemm_nnz, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);

        select_bad_arg_analysis(rocsparse_csrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);

        temp_buffer = (void*)0x4;
    }

    // ###############################################
    // Scenario 4: alpha != nullptr && beta != nullptr
    // ###############################################

    {
        // In this scenario matrices A != B != D != nullptr
        int nargs_to_exclude_buffer_size = 2;
        int nargs_to_exclude             = 2;

        const int args_to_exclude_buffer_size[2] = {6, 15};
        const int args_to_exclude[2]             = {6, 17};

        const T* alpha = &h_alpha;
        const T* beta  = &h_beta;

        // A matrix
        const rocsparse_mat_descr descr_A       = local_descr_A;
        rocsparse_int             nnz_A         = safe_size;
        const T*                  csr_val_A     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)0x4;

        // B matrix
        const rocsparse_mat_descr descr_B       = local_descr_B;
        rocsparse_int             nnz_B         = safe_size;
        const T*                  csr_val_B     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)0x4;

        // D matrix
        const rocsparse_mat_descr descr_D       = local_descr_D;
        rocsparse_int             nnz_D         = safe_size;
        const T*                  csr_val_D     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_D = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_D = (const rocsparse_int*)0x4;

        select_bad_arg_analysis(rocsparse_csrgemm_buffer_size<T>,
                                nargs_to_exclude_buffer_size,
                                args_to_exclude_buffer_size,
                                PARAMS_BUFFER_SIZE);
        bad_arg_analysis(rocsparse_csrgemm_nnz, PARAMS_NNZ);
        select_bad_arg_analysis(rocsparse_csrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    //
    // Not implemented cases.
    //

    {
        const T* alpha = &h_alpha;
        const T* beta  = &h_beta;
        // A matrix
        const rocsparse_mat_descr descr_A       = local_descr_A;
        rocsparse_int             nnz_A         = safe_size;
        const T*                  csr_val_A     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)0x4;

        // B matrix
        const rocsparse_mat_descr descr_B       = local_descr_B;
        rocsparse_int             nnz_B         = safe_size;
        const T*                  csr_val_B     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)0x4;

        // D matrix
        const rocsparse_mat_descr descr_D       = local_descr_D;
        rocsparse_int             nnz_D         = safe_size;
        const T*                  csr_val_D     = (const T*)0x4;
        const rocsparse_int*      csr_row_ptr_D = (const rocsparse_int*)0x4;
        const rocsparse_int*      csr_col_ind_D = (const rocsparse_int*)0x4;

        {
            rocsparse_operation op = trans_A;
            trans_A                = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            trans_A = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            trans_A = op;

            op      = trans_B;
            trans_B = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            trans_B = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            trans_B = op;
        }

        {
            rocsparse_operation op = trans_A;
            trans_A                = rocsparse_operation_transpose;

            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(PARAMS_NNZ),
                                    rocsparse_status_not_implemented);

            trans_A = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(PARAMS_NNZ),
                                    rocsparse_status_not_implemented);
            trans_A = op;

            op      = trans_B;
            trans_B = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(PARAMS_NNZ),
                                    rocsparse_status_not_implemented);
            trans_B = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(PARAMS_NNZ),
                                    rocsparse_status_not_implemented);
            trans_B = op;
        }

        {
            rocsparse_operation op = trans_A;
            trans_A                = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(PARAMS), rocsparse_status_not_implemented);
            trans_A = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(PARAMS), rocsparse_status_not_implemented);
            trans_A = op;

            op      = trans_B;
            trans_B = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(PARAMS), rocsparse_status_not_implemented);
            trans_B = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(PARAMS), rocsparse_status_not_implemented);
            trans_B = op;
        }
    }

#undef PARAMS
#undef PARAMS_NNZ
#undef PARAMS_BUFFER_SIZE
}

enum testing_csrgemm_scenario
{
    testing_csrgemm_scenario_none,
    testing_csrgemm_scenario_alpha,
    testing_csrgemm_scenario_beta,
    testing_csrgemm_scenario_alpha_and_beta,
};

template <typename T>
void testing_csrgemm(const Arguments& arg)
{
    std::cout << "random_cached_generator<float>(): " << random_cached_generator<float>()
              << std::endl;
    rocsparse_seedrand();

    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_operation   transA    = arg.transA;
    rocsparse_operation   transB    = arg.transB;
    rocsparse_index_base  baseA     = arg.baseA;
    rocsparse_index_base  baseB     = arg.baseB;
    rocsparse_index_base  baseC     = arg.baseC;
    rocsparse_index_base  baseD     = arg.baseD;
    static constexpr bool full_rank = false;

    T v_alpha = arg.get_alpha<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descrA;
    rocsparse_local_mat_descr descrB;
    rocsparse_local_mat_descr descrC;
    rocsparse_local_mat_descr descrD;
    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrC, baseC));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrD, baseD));

    // Create matrix info for C
    rocsparse_local_mat_info info;

    void* dbuffer = nullptr;

#define PARAMS_BUFFER_SIZE(alpha, beta, A, B, C, D, out_buffer_size)                          \
    handle, transA, transB, A.m, C.n, A.n, alpha, descrA, A.nnz, A.ptr, A.ind, descrB, B.nnz, \
        B.ptr, B.ind, beta, descrD, D.nnz, D.ptr, D.ind, info, &out_buffer_size

#define PARAMS_NNZ(A, B, C, D, out_nnz)                                                       \
    handle, transA, transB, A.m, C.n, A.n, descrA, A.nnz, A.ptr, A.ind, descrB, B.nnz, B.ptr, \
        B.ind, descrD, D.nnz, D.ptr, D.ind, descrC, C.ptr, out_nnz, info, dbuffer

#define PARAMS(alpha, beta, A, B, C, D)                                                       \
    handle, transA, transB, A.m, C.n, A.n, alpha, descrA, A.nnz, A.val, A.ptr, A.ind, descrB, \
        B.nnz, B.val, B.ptr, B.ind, beta, descrD, D.nnz, D.val, D.ptr, D.ind, descrC, C.val,  \
        C.ptr, C.ind, info, dbuffer

    host_dense_vector<T> h_alpha(0), h_beta(0);
    h_alpha.resize(1);
    *h_alpha = v_alpha;

    // Declare host objects.
    host_csr_matrix<T> h_A, h_B, h_C, h_D;

    // Initialize matrices.
    {
        rocsparse_matrix_factory<T> matrix_factory(arg, arg.timing ? false : true, full_rank);
        matrix_factory.init_csr(h_A, M, K, baseA);

        rocsparse_matrix_factory_random<T> rf(full_rank);
        {
            h_B.base = baseB;
            h_B.m    = K;
            h_B.n    = N;
            rf.init_csr(h_B.ptr,
                        h_B.ind,
                        h_B.val,
                        h_B.m,
                        h_B.n,
                        h_B.nnz,
                        h_B.base,
                        rocsparse_matrix_type_general,
                        rocsparse_fill_mode_lower,
                        rocsparse_storage_mode_sorted);
        }

        h_C.define(M, N, 0, baseC);
    }

    std::cout << "staring A ptr" << std::endl;
    for(size_t i = 0; i < h_A.ptr.size(); i++)
    {
        std::cout << h_A.ptr[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "staring A ind" << std::endl;
    for(size_t i = 0; i < h_A.ind.size(); i++)
    {
        std::cout << h_A.ind[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "staring A val" << std::endl;
    for(size_t i = 0; i < std::min(h_A.val.size(), (size_t)10); i++)
    {
        std::cout << h_A.val[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "staring B ptr" << std::endl;
    for(size_t i = 0; i < h_B.ptr.size(); i++)
    {
        std::cout << h_B.ptr[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "staring B ind" << std::endl;
    for(size_t i = 0; i < h_B.ind.size(); i++)
    {
        std::cout << h_B.ind[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "staring B val" << std::endl;
    for(size_t i = 0; i < std::min(h_B.val.size(), (size_t)10); i++)
    {
        std::cout << h_B.val[i] << " ";
    }
    std::cout << "" << std::endl;

    // h_A.define(M, K, M * K, baseA);
    // h_B.define(K, N, K * N, baseB);
    // h_C.define(M, N, 0, baseC);

    // h_A.ptr[0] = 0;
    // for(int i = 0; i < M; i++)
    // {
    //     h_A.ptr[i + 1] = h_A.ptr[i] + K;
    // }

    // for(int i = 0; i < M; i++)
    // {
    //     int start = h_A.ptr[i] - baseA;
    //     int end   = h_A.ptr[i + 1] - baseA;

    //     for(int j = start; j < end; j++)
    //     {
    //         h_A.ind[j] = j - start + baseA;
    //     }
    // }

    // h_B.ptr[0] = 0;
    // for(int i = 0; i < K; i++)
    // {
    //     h_B.ptr[i + 1] = h_B.ptr[i] + N;
    // }

    // for(int i = 0; i < K; i++)
    // {
    //     int start = h_B.ptr[i] - baseB;
    //     int end   = h_B.ptr[i + 1] - baseB;

    //     for(int j = start; j < end; j++)
    //     {
    //         h_B.ind[j] = j - start + baseB;
    //     }
    // }

    // std::cout << "h_A" << std::endl;
    // for(int i = 0; i < M; i++)
    // {
    //     int start = h_A.ptr[i] - baseA;
    //     int end   = h_A.ptr[i + 1] - baseA;

    //     std::vector<T> htemp(K, 0);
    //     for(int j = start; j < end; j++)
    //     {
    //         htemp[h_A.ind[j] - baseA] = 1;
    //     }

    //     for(int j = 0; j < K; j++)
    //     {
    //         std::cout << htemp[j] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }
    // std::cout << "" << std::endl;

    // std::cout << "h_B" << std::endl;
    // for(int i = 0; i < K; i++)
    // {
    //     int start = h_B.ptr[i] - baseB;
    //     int end   = h_B.ptr[i + 1] - baseB;

    //     std::vector<T> htemp(N, 0);
    //     for(int j = start; j < end; j++)
    //     {
    //         htemp[h_B.ind[j] - baseB] = 1;
    //     }

    //     for(int j = 0; j < N; j++)
    //     {
    //         std::cout << htemp[j] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }
    // std::cout << "" << std::endl;

    // Declare device objects.
    device_csr_matrix<T>   d_A(h_A), d_B(h_B), d_C(h_C), d_D(h_D);
    device_dense_vector<T> d_alpha(h_alpha), d_beta(h_beta);

    // Obtain required buffer size
    size_t out_buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_buffer_size<T>(
        PARAMS_BUFFER_SIZE(h_alpha, h_beta, d_A, d_B, d_C, d_D, out_buffer_size)));

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, out_buffer_size));

    if(arg.unit_check)
    {
        // Host calculation.
        {
            rocsparse_int out_nnz;

            host_csrgemm_nnz<T, rocsparse_int, rocsparse_int>(h_A.m,
                                                              h_C.n,
                                                              h_A.n,
                                                              h_alpha,
                                                              h_A.ptr,
                                                              h_A.ind,
                                                              h_B.ptr,
                                                              h_B.ind,
                                                              h_beta,
                                                              h_D.ptr,
                                                              h_D.ind,
                                                              h_C.ptr,
                                                              &out_nnz,
                                                              h_A.base,
                                                              h_B.base,
                                                              h_C.base,
                                                              h_D.base);

            h_C.define(h_C.m, h_C.n, out_nnz, h_C.base);

            host_csrgemm<T, rocsparse_int, rocsparse_int>(h_A.m,
                                                          h_C.n,
                                                          h_A.n,
                                                          h_alpha,
                                                          h_A.ptr,
                                                          h_A.ind,
                                                          h_A.val,
                                                          h_B.ptr,
                                                          h_B.ind,
                                                          h_B.val,
                                                          h_beta,
                                                          h_D.ptr,
                                                          h_D.ind,
                                                          h_D.val,
                                                          h_C.ptr,
                                                          h_C.ind,
                                                          h_C.val,
                                                          h_A.base,
                                                          h_B.base,
                                                          h_C.base,
                                                          h_D.base);
        }

        {
            // GPU with pointer mode host
            host_scalar<rocsparse_int> h_out_nnz;
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(PARAMS_NNZ(d_A, d_B, d_C, d_D, h_out_nnz)));
            // d_C.define(d_C.m, d_C.n, *h_out_nnz, d_C.base);
            // CHECK_ROCSPARSE_ERROR(
            //     rocsparse_csrgemm<T>(PARAMS(h_alpha, h_beta, d_A, d_B, d_C, d_D)));
            // if(ROCSPARSE_REPRODUCIBILITY)
            // {
            //     rocsparse_reproducibility::save("d_C pointer mode host", d_C);
            // }

            // h_C.near_check(d_C);
        }
    }

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csrgemm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrgemm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrgemm_extra(const Arguments& arg) {}
