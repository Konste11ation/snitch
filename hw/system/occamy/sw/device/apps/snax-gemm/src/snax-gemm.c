// Copyright 2023 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
#include "snrt.h"
#include "data.h"

#define tileSize 8
#define meshRow 8
#define meshCol 8
// This tests the following:
// 1) Generate random data with gendata.py
// 2) Allocate space in TCDM
// 3) Write data from L3 to TCDM
// 4) Configure the csrs for performing a block GEMM
// 5) Launch the accelerator
// 6) Wait until the accelerator finishes
// 7) Check the result of the CPU and the accelerator vs the golden model
// (gendata.py)

int32_t gen_size_config(uint8_t Batch, uint8_t M, uint8_t K, uint8_t N) {
    return ((int32_t)Batch << 24) | ((int32_t)M << 16) | ((int32_t)K << 8) |
           (int32_t)N;
}
int32_t gen_subtraction_config(int8_t subtraction_a, int8_t subtraction_b) {
    return ((uint8_t)subtraction_b << 8) | (uint8_t)subtraction_a;
}
uint32_t read_performance_counter() {
    uint32_t performance_counter;
    performance_counter = read_csr(0x3cd);
    return performance_counter;
};
void load_input_data(uint8_t Batch, uint8_t M, uint8_t K, uint8_t N,
                     int8_t* local_a, int8_t* local_b, int8_t* A, int8_t* B,
                     uint32_t strideInnermostA, uint32_t strideInnermostB,
                     uint32_t ldA, uint32_t ldB, uint32_t strideA,
                     uint32_t strideB) {
    int8_t* addr_a;
    int8_t* addr_b;

    int8_t* addr_A;
    int8_t* addr_B;

    for (int b = 0; b < Batch; b++) {
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                // generate the start address of sub-matrix of A in TCDM
                // according to the strides definition
                addr_a =
                    local_a + (b * strideA + m * ldA + k * strideInnermostA) /
                                  sizeof(int8_t);
                // element index of A
                addr_A =
                    A + (b * M * meshRow * tileSize * K +
                         m * meshRow * tileSize * K + k * meshRow * tileSize) /
                            sizeof(int8_t);
                snrt_dma_start_1d(addr_a, addr_A,
                                  meshRow * tileSize * sizeof(int8_t));
            }
        }
    }

    for (int b = 0; b < Batch; b++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                // generate the start address of sub-matrix of B in TCDM
                // according to the strides definition
                addr_b =
                    local_b + (b * strideB + n * ldB + k * strideInnermostB) /
                                  sizeof(int8_t);
                // element index of B
                addr_B =
                    B + (b * K * tileSize * meshCol * N +
                         n * tileSize * meshCol * K + k * tileSize * meshCol) /
                            sizeof(int8_t);
                snrt_dma_start_1d(addr_b, addr_B,
                                  meshCol * tileSize * sizeof(int8_t));
            }
        }
    }
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


int main() {

    // Prepare addresses in TCDM
    int8_t *local_a, *local_b;
    int32_t* local_c;

    // Allocate space in TCDM
    local_a = (int8_t*)snrt_l1_next();
    local_b = local_a + delta_local_a * sizeof(int8_t);
    local_c = (int32_t*)(local_b + delta_local_b * sizeof(int8_t));

    uint32_t dma_pre_load = mcycle();

    // Transfer data from L3 to L1
    // Using DMA only
    if (snrt_is_dm_core()) {
        load_input_data(Batch, M, K, N, local_a, local_b, A, B,
                        strideInnermostA, strideInnermostB, ldA, ldB, strideA,
                        strideB);
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        // Pack matrix size setting to one CSR
        uint32_t size_setting = gen_size_config(Batch, M, K, N);

        uint32_t subtraction_setting =
            gen_subtraction_config(subtraction_a, subtraction_b);

        uint32_t gemm_start = mcycle();

        // Set GEMM configuration CSR
        set_batch_gemm(size_setting, local_a, local_b, subtraction_setting,
                       local_c, strideInnermostA, strideInnermostB,
                       strideInnermostC, ldA, ldB, ldC, strideA, strideB,
                       strideC);

        // Set CSR to start GEMM and poll until GEMM accelerator finishes
        start_batch_gemm();
        wait_batch_gemm();

        uint32_t gemm_end = mcycle();
    };

    snrt_cluster_hw_barrier();

    return_to_cva6(SYNC_ALL);
}