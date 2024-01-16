#include "stdint.h"
#pragma once

/**
 * @struct snax_gemm_layer_struct
 * @brief This structure contains all parameters necessary for snax_gemm.
 * @var snax_gemm_layer_struct::Batch
 * Number of Batch
 * @var snax_gemm_layer_struct::M
 * Dimension of A matrix = M*Mu x K*Ku. How many rows of Mu.
 * @var snax_gemm_layer_struct::K
 * Dimension of A matrix = M*Mu x K*Ku. How many cols of Ku.
 * @var snax_gemm_layer_struct::N
 * Dimension of B matrix = K*Ku x N*Nu. How many cols of Nu.
 * @var snax_gemm_layer_struct::*A
 * Matrix A start address
 * @var snax_gemm_layer_struct::*B
 * Matrix B start address
 * @var snax_gemm_layer_struct::subtraction_a
 * Scalar Value. The input to GeMM is (A-a)
 * @var snax_gemm_layer_struct::subtraction_b
 * Scalar Value. The input to GeMM is (B-b)
 * @var snax_gemm_layer_struct::*C
 * Matrix C start address
 * @var snax_gemm_layer_struct::strideInnermostA
 * The incremental address for a new submatrix of each block row of A
 * @var snax_gemm_layer_struct::strideInnermostB
 * the incremental address for a new submatrix of each block column of B
 * @var snax_gemm_layer_struct::strideInnermostC
 * the incremental address for a new submatrix of each block row of C
 * @var snax_gemm_layer_struct::ldA
 * the incremental address for a new block row of A
 * @var snax_gemm_layer_struct::ldB
 * the incremental address for a new block column of B
 * @var snax_gemm_layer_struct::ldC
 * the incremental address for a new block row of C
 * @var snax_gemm_layer_struct::strideA
 * the incremental address of matrix A for each batch
 * @var snax_gemm_layer_struct::strideB
 * the incremental address of matrix B for each batch
 * @var snax_gemm_layer_struct::strideC
 * the incremental address of matrix C for each batch
 */
typedef struct snax_gemm_layer_struct {
    uint8_t Batch;
    uint8_t M;
    uint8_t K;
    uint8_t N;

    int8_t *A;
    int8_t *B;

    int8_t subtraction_a;
    int8_t subtraction_b;

    int32_t *C;

    int32_t strideInnermostA;
    int32_t strideInnermostB;
    int32_t strideInnermostC;
    
    int32_t ldA;
    int32_t ldB;
    int32_t ldC;

    int32_t strideA;
    int32_t strideB;
    int32_t strideC;

} snax_gemm_layer;