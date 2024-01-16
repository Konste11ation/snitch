// Copyright 2023 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include "snax-gemm-lib.h"

#include "snax-gemm-params.h"

uint32_t gen_size_config(uint8_t Batch, uint8_t M, uint8_t K, uint8_t N) {
    return ((uint32_t)Batch << 24) | ((uint32_t)M << 16) | ((uint32_t)K << 8) |
           (uint32_t)N;
}

int32_t gen_subtraction_config(int8_t subtraction_a, int8_t subtraction_b) {
    return ((uint8_t)subtraction_b << 8) | (uint8_t)subtraction_a;
}



uint32_t read_performance_counter() {
    uint32_t performance_counter;
    performance_counter = read_csr(0x3cd);
    return performance_counter;
};

void base_gemm(uint8_t m, uint8_t k, uint8_t n, int8_t* A, int8_t* B,
               int8_t subtraction_a, int8_t subtraction_b, int32_t* C_cpu,
               bool clear) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // clear memory first before start matrix multiplication
            // to accumulate in K dimension
            if (clear == true) {
                C_cpu[i * n + j] = 0;
            }
            for (int s = 0; s < k; s++) {
                C_cpu[i * n + j] =
                    C_cpu[i * n + j] +
                    ((int32_t)A[i * k + s] - (int32_t)subtraction_a) *
                        ((int32_t)B[s + j * k] - (int32_t)subtraction_b);
            }
        }
    }
};

void batch_gemm_cpu(uint8_t Batch, uint8_t M, uint8_t K, uint8_t N, int8_t* A,
                    int8_t* B, int8_t subtraction_a, int8_t subtraction_b,
                    int32_t* C, uint32_t strideInnermostA,
                    uint32_t strideInnermostB, uint32_t strideInnermostC,
                    uint32_t ldA, uint32_t ldB, uint32_t ldC, uint32_t strideA,
                    uint32_t strideB, uint32_t strideC) {
    int8_t* start_addr_a = A;
    int8_t* start_addr_b = B;
    int32_t* start_addr_c = C;
    int8_t* addr_a;
    int8_t* addr_b;
    int32_t* addr_c;

    bool clear;

    for (int b = 0; b < Batch; b++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    // generate the start address of each sub-matrix
                    // according to the strides definition
                    addr_a = start_addr_a +
                             (b * strideA + m * ldA + k * strideInnermostA) /
                                 sizeof(int8_t);
                    addr_b = start_addr_b +
                             (b * strideB + n * ldB + k * strideInnermostB) /
                                 sizeof(int8_t);
                    addr_c = start_addr_c +
                             (b * strideC + m * ldC + n * strideInnermostC) /
                                 sizeof(int32_t);
                    // when k == 0, clear the memory
                    clear = k == 0;
                    base_gemm(meshRow, tileSize, meshCol, addr_a, addr_b,
                              subtraction_a, subtraction_b, addr_c, clear);
                }
            }
        }
    }
}

uint32_t get_delta_addr_a(uint8_t b,
                          uint8_t m, uint8_t k,
                          uint32_t strideInnermostA,
                          uint32_t ldA,
                          uint32_t strideA){
    uint32_t delta_addr_a;

    delta_addr_a = (b * strideA + m * ldA + k * strideInnermostA) / sizeof(int8_t);
    return delta_addr_a;
}

uint32_t get_delta_addr_b(uint8_t b,
                          uint8_t n, uint8_t k,
                          uint32_t strideInnermostB,
                          uint32_t ldB,
                          uint32_t strideB){
    uint32_t delta_addr_b;

    delta_addr_b = (b * strideB + n * ldB + k * strideInnermostB) / sizeof(int8_t);
    return delta_addr_b;
}

uint32_t get_delta_addr_A(uint8_t b,
                          uint8_t M, uint8_t K,
                          uint8_t m, uint8_t k){
    uint32_t delta_addr_A;

    delta_addr_A = (b * M * meshRow * tileSize * K + m * meshRow * tileSize * K + k * meshRow * tileSize) / sizeof(int8_t);  
    return delta_addr_A;
}

uint32_t get_delta_addr_B(uint8_t b,
                          uint8_t N, uint8_t K,
                          uint8_t n, uint8_t k){
    uint32_t delta_addr_B;

    delta_addr_B = (b * K * tileSize * meshCol * N + n * tileSize * meshCol * K + k * tileSize * meshCol) / sizeof(int8_t);  
    return delta_addr_B;
}




void load_input_A_block(uint8_t b,
                        uint8_t m, uint8_t k,
                        uint8_t M, uint8_t K,
                        int8_t* local_a, 
                        int8_t* A,
                        uint32_t strideInnermostA,
                        uint32_t ldA, 
                        uint32_t strideA){
    int8_t* addr_a;
    int8_t* addr_A;
    // generate the start address of sub-matrix of A in TCDM
    // according to the strides definition  
    // uint32_t delta_addr_a;
    // uint32_t delta_addr_A;
    // delta_addr_a = get_delta_addr_a(b,
    //                                 m,k,
    //                                 strideInnermostA,
    //                                 ldA,
    //                                 strideA);
    // delta_addr_A = get_delta_addr_A(b,
    //                                 M,K,
    //                                 m,k);
    addr_a = local_a + (b * strideA + m * ldA + k * strideInnermostA) / sizeof(int8_t);
    //addr_a = local_a + delta_addr_a;
    // element index of A
    //addr_A = A + delta_addr_A;
    addr_A = A + (b * M * meshRow * tileSize * K +
                  m * meshRow * tileSize * K + k * meshRow * tileSize) / sizeof(int8_t);  
    snrt_dma_start_1d(addr_a, addr_A, meshRow * tileSize * sizeof(int8_t));    
}


void load_input_B_block(uint8_t b,
                        uint8_t n, uint8_t k,
                        uint8_t N, uint8_t K,
                        int8_t* local_b, 
                        int8_t* B,
                        uint32_t strideInnermostB,
                        uint32_t ldB, 
                        uint32_t strideB){
    int8_t* addr_b;
    int8_t* addr_B; 
    // uint32_t delta_addr_b;
    // uint32_t delta_addr_B;
    // delta_addr_b = get_delta_addr_b(b,
    //                                 n,k,
    //                                 strideInnermostB,
    //                                 ldB,
    //                                 strideB); 
    // delta_addr_B = get_delta_addr_B(b,
    //                                 N,K,
    //                                 n,k);  
    // generate the start address of sub-matrix of B in TCDM
    // according to the strides definition
    //addr_b = local_b + delta_addr_b;
    addr_b = local_b + (b * strideB + n * ldB + k * strideInnermostB) / sizeof(int8_t);
    // element index of B
    //addr_B = B + delta_addr_B;
    addr_B = B + (b * K * tileSize * meshCol * N + n * tileSize * meshCol * K + k * tileSize * meshCol) / sizeof(int8_t);
    snrt_dma_start_1d(addr_b, addr_B, meshCol * tileSize * sizeof(int8_t));                     
}


void load_input_A(uint8_t Batch, 
                  uint8_t M, uint8_t K,
                  int8_t* local_a, 
                  int8_t* A,
                  uint32_t strideInnermostA,
                  uint32_t ldA, 
                  uint32_t strideA) {
    
    for (uint8_t b = 0; b < Batch; b++) {
         for (uint8_t m = 0; m < M; m++) {
            for (uint8_t k = 0; k < K; k++) {
                load_input_A_block(b,
                                   m,k,
                                   M,K,
                                   local_a,
                                   A,
                                   strideInnermostA,
                                   ldA,
                                   strideA);
            }
        }
    }
    // int8_t* addr_a;

    // int8_t* addr_A;
    // for (int b = 0; b < Batch; b++) {
    //     for (int m = 0; m < M; m++) {
    //         for (int k = 0; k < K; k++) {
    //             // generate the start address of sub-matrix of A in TCDM
    //             // according to the strides definition
    //             addr_a =
    //                 local_a + (b * strideA + m * ldA + k * strideInnermostA) /
    //                               sizeof(int8_t);
    //             // element index of A
    //             addr_A =
    //                 A + (b * M * meshRow * tileSize * K +
    //                      m * meshRow * tileSize * K + k * meshRow * tileSize) /
    //                         sizeof(int8_t);
    //             snrt_dma_start_1d(addr_a, addr_A,
    //                               meshRow * tileSize * sizeof(int8_t));
    //         }
    //     }
    // }
}

void load_input_B(uint8_t Batch, 
                  uint8_t N, uint8_t K,
                  int8_t* local_b, 
                  int8_t* B,
                  uint32_t strideInnermostB,
                  uint32_t ldB, 
                  uint32_t strideB) {
    for (uint8_t b = 0; b < Batch; b++) {
        for (uint8_t n = 0; n < N; n++) {
            for (uint8_t k = 0; k < K; k++) {
                load_input_B_block(b,
                                   n,k,
                                   N,K,
                                   local_b,
                                   B,
                                   strideInnermostB,
                                   ldB,
                                   strideB);                
            }
        }
    }   
    // int8_t* addr_b;

    // int8_t* addr_B;
    // for (int b = 0; b < Batch; b++) {
    //     for (int n = 0; n < N; n++) {
    //         for (int k = 0; k < K; k++) {
    //             // generate the start address of sub-matrix of B in TCDM
    //             // according to the strides definition
    //             addr_b =
    //                 local_b + (b * strideB + n * ldB + k * strideInnermostB) /
    //                               sizeof(int8_t);
    //             // element index of B
    //             addr_B =
    //                 B + (b * K * tileSize * meshCol * N +
    //                      n * tileSize * meshCol * K + k * tileSize * meshCol) /
    //                         sizeof(int8_t);
    //             snrt_dma_start_1d(addr_b, addr_B,
    //                               meshCol * tileSize * sizeof(int8_t));
    //         }
    //     }
    // }
}

void load_input_data(uint8_t Batch, uint8_t M, uint8_t K, uint8_t N,
                     int8_t* local_a, int8_t* local_b, int8_t* A, int8_t* B,
                     uint32_t strideInnermostA, uint32_t strideInnermostB,
                     uint32_t ldA, uint32_t ldB, uint32_t strideA,
                     uint32_t strideB) {

    load_input_A(Batch,
                 M, K,
                 local_a,
                 A,
                 strideInnermostA,
                 ldA,
                 strideA);
    load_input_B(Batch,
                 N, K,
                 local_b,
                 B,
                 strideInnermostB,
                 ldB,
                 strideB);
    // int8_t* addr_a;
    // int8_t* addr_b;

    // int8_t* addr_A;
    // int8_t* addr_B;

    // for (int b = 0; b < Batch; b++) {
    //     for (int m = 0; m < M; m++) {
    //         for (int k = 0; k < K; k++) {
    //             // generate the start address of sub-matrix of A in TCDM
    //             // according to the strides definition
    //             addr_a =
    //                 local_a + (b * strideA + m * ldA + k * strideInnermostA) /
    //                               sizeof(int8_t);
    //             // element index of A
    //             addr_A =
    //                 A + (b * M * meshRow * tileSize * K +
    //                      m * meshRow * tileSize * K + k * meshRow * tileSize) /
    //                         sizeof(int8_t);
    //             snrt_dma_start_1d(addr_a, addr_A,
    //                               meshRow * tileSize * sizeof(int8_t));
    //         }
    //     }
    // }

    // for (int b = 0; b < Batch; b++) {
    //     for (int n = 0; n < N; n++) {
    //         for (int k = 0; k < K; k++) {
    //             // generate the start address of sub-matrix of B in TCDM
    //             // according to the strides definition
    //             addr_b =
    //                 local_b + (b * strideB + n * ldB + k * strideInnermostB) /
    //                               sizeof(int8_t);
    //             // element index of B
    //             addr_B =
    //                 B + (b * K * tileSize * meshCol * N +
    //                      n * tileSize * meshCol * K + k * tileSize * meshCol) /
    //                         sizeof(int8_t);
    //             snrt_dma_start_1d(addr_b, addr_B,
    //                               meshCol * tileSize * sizeof(int8_t));
    //         }
    //     }
    // }
}


void set_batch_gemm(uint32_t size_setting, int8_t* local_a, int8_t* local_b,
                    int32_t subtractions, int32_t* local_c,
                    uint32_t strideInnermostA, uint32_t strideInnermostB,
                    uint32_t strideInnermostC, uint32_t ldA, uint32_t ldB,
                    uint32_t ldC, uint32_t strideA, uint32_t strideB,
                    uint32_t strideC) {
    // Set matrix size
    write_csr(0x3c0, size_setting);

    // Set addresses
    write_csr(0x3c1, (uint32_t)local_a);
    write_csr(0x3c2, (uint32_t)local_b);
    write_csr(0x3c3, (uint32_t)local_c);

    // Set strides
    write_csr(0x3c4, strideInnermostA);
    write_csr(0x3c5, strideInnermostB);
    write_csr(0x3c6, strideInnermostC);

    write_csr(0x3c7, ldA);
    write_csr(0x3c8, ldB);
    write_csr(0x3c9, ldC);

    write_csr(0x3ca, strideA);
    write_csr(0x3cb, strideB);
    write_csr(0x3cc, strideC);

    // Set subtraction values
    write_csr(0x3ce, subtractions);
}


void start_batch_gemm() {
    // 0x3ce is the CSR address for accelerator status
    // set the lowest bit of state CSR  (state CSR[0]) to set start signal in
    // GEMM
    write_csr(0x3cf, 1);
}

void wait_batch_gemm() {
    uint32_t break_poll;

    while (1) {
        // poll the state CSR[1] to see if GEMM is still busy
        break_poll = read_csr(0x3cf);
        if ((break_poll >> 1) == 1) {
            break;
        };
    };
}

uint32_t check_result(int32_t* output, int32_t* output_golden, uint8_t Batch,
                      uint8_t M, uint8_t N, uint32_t strideInnermostC,
                      uint32_t ldC, uint32_t strideC) {
    /*
     * Compare output to output_golden with length
     */
    uint32_t err = 0;
    uint32_t golden_idx;
    int32_t* out_addr;

    for (int b = 0; b < Batch; b++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                for (int i = 0; i < meshRow; i++) {
                    for (int j = 0; j < meshCol; j++) {
                        // element index of golden results
                        golden_idx =
                            i * meshCol + j + b * M * meshRow * meshCol * N +
                            m * meshRow * meshCol * N + n * meshRow * meshCol;
                        // generate the start address of
                        // sub-matrix of output C in TCDM
                        // according to the strides definition
                        out_addr =
                            output +
                            (b * strideC + m * ldC + n * strideInnermostC) /
                                sizeof(int32_t) +
                            i * meshCol + j;
                        // Check if output is same as golden output
                        if ((int32_t)*out_addr != output_golden[golden_idx]) {
                            err++;
                        };
                    };
                }
            }
        }
    }

    return err;
}