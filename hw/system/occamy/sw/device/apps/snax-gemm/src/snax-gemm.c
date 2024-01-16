// Copyright 2023 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
#include "snrt.h"
#include "data.h"


int main() {
    // Perform some operations (e.g. clear interrupt) after wakeup
    post_wakeup_cl();
    // Core            hard_id   Global Core idx = hartid - base_hardid = hartid - 1
    // ---------------------------
    // CVA6            0
    // ---------------------------
    // Group0:SNAX     1         0
    // Group0:DM Core  2         1
    // ---------------------------
    // Group1:SNAX     3         2
    // Group1:DM Core  4         3
    // ---------------------------
    // Group2:SNAX     5         4
    // Group2:DM Core  6         5
    // ---------------------------
    // Group3:SNAX     7         6
    // Group3:DM Core  8         7
    // ---------------------------
    uint32_t group0_bound_lower = 0;
    uint32_t group0_bound_upper = snrt_cluster_core_num(); // 2
    uint32_t group1_bound_lower = group0_bound_upper;
    uint32_t group1_bound_upper = 2*snrt_cluster_core_num(); // 4
    uint32_t group2_bound_lower = group1_bound_upper;
    uint32_t group2_bound_upper = 3*snrt_cluster_core_num(); // 6
    uint32_t group3_bound_lower = group2_bound_upper;
    uint32_t group3_bound_upper = 4*snrt_cluster_core_num(); // 8

    uint8_t *local_a, *local_b;
    uint32_t *local_c;
    uint32_t *local_err;
    uint32_t tic, toc;

    if(snrt_global_core_idx()<group0_bound_upper){
        // Allocate space in TCDM
        local_a = (uint8_t *)snrt_l1_next();
        local_b = local_a + k * n * sizeof(uint8_t);
        local_c = (uint32_t *)(local_b + m * n * sizeof(uint8_t));
        local_err = (uint32_t *)(local_c + m * n * sizeof(uint8_t));

        // Transfer data from L3 to L1
        // Using DMA only
        if (snrt_is_dm_core()) {
            tic = mcycle();

            snrt_dma_start_1d(local_a, A, m * k * sizeof(uint8_t));
            snrt_dma_start_1d(local_b, B, n * k * sizeof(uint8_t));
            toc = mcycle();
        }

        // Wait for DMA to finish
        snrt_cluster_hw_barrier();

        // Base MM calculation
        if (snrt_is_compute_core()) {
            uint32_t temp_accumulator;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    temp_accumulator = 0;
                    for (int s = 0; s < k; s++) {
                        temp_accumulator += (uint32_t)(*(local_a + i * k + s)) *
                                            (uint32_t)(*(local_b + s + j * k));
                    }
                    *(local_c + i * k + j) = temp_accumulator;
                }
            }
        }
    }
    snrt_global_barrier();
    // Check results
    if(snrt_global_core_idx()<group0_bound_upper){
        if (snrt_is_compute_core()) {
            // Check if result is not equal to golden result
            for (uint32_t i = 0; i < m; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    if (C_golden[i * n + j] != *(local_c + (i * n + j))) {
                        *(local_err + (i * n + j )) = 1;
                    };
                }
            }
        }
        // Wait for Check to finish
        snrt_cluster_hw_barrier();
        // Transfer the error signal out to the main memory
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ERROR, local_err, m * n * sizeof(uint8_t));
        }
    }

    snrt_global_barrier();
    return_to_cva6(SYNC_ALL);
}