// Copyright 2023 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
#include "data.h"

#include "snax-gemm-lib.h"
#include "snax-gemm-params.h"

void perform_snax_gemm (int8_t *local_a, int8_t *local_b, int32_t *local_c,
                        uint8_t Batch,
                        uint8_t M, uint8_t K, uint8_t N,
                        int8_t *A, int8_t *B,
                        int8_t subtraction_a, int8_t subtraction_b,
                        int32_t strideInnermostA, int32_t strideInnermostB, int32_t strideInnermostC,
                        int32_t ldA, int32_t ldB, int32_t ldC,
                        int32_t strideA, int32_t strideB, int32_t strideC){
        
        // Transfer data from L3 to L1
        // Using DMA only
        if (snrt_is_dm_core()) {
            uint32_t dma_pre_load_start = mcycle();
            load_input_data(Batch, 
                            M, K, N, 
                            local_a, local_b, 
                            A, B,
                            strideInnermostA, strideInnermostB, 
                            ldA, ldB, 
                            strideA, strideB);
            snrt_dma_wait_all();    
            uint32_t dma_pre_load_end = mcycle();
        }
        
        // Wait for DMA to finish
        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core()) {
            // Pack matrix size setting to one CSR
            uint32_t size_setting = gen_size_config(Batch, 
                                                    M, K, N);

            uint32_t subtraction_setting =
                gen_subtraction_config(subtraction_a, subtraction_b);
            uint32_t gemm_start = mcycle();

            // Set GEMM configuration CSR
            set_batch_gemm(size_setting, 
                           local_a, local_b, 
                           subtraction_setting,
                           local_c, 
                           strideInnermostA, 
                           strideInnermostB,
                           strideInnermostC, 
                           ldA, 
                           ldB, 
                           ldC, 
                           strideA, 
                           strideB,
                           strideC);
            start_batch_gemm();
            wait_batch_gemm();

            uint32_t gemm_end = mcycle();
        };

}

int main() {
    post_wakeup_cl();
    // Set err value for checking
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
    // Prepare addresses in TCDM
    uint32_t *local_err;
    int8_t *local_a, *local_b;
    int32_t *local_c;
    

    // First do the 16x16x16 GeMM
    if(snrt_global_core_idx()<group0_bound_upper){
        local_err = (uint32_t*)snrt_l1_next();
        local_a = (int8_t*)(local_err+ 256);
        local_b = local_a + delta_local_a * sizeof(int8_t);
        local_c = (int32_t*)(local_b + delta_local_b * sizeof(int8_t));
        perform_snax_gemm(local_a, local_b, local_c, 
                            1,
                            2, 2, 2,
                            A, B,
                            subtraction_a, subtraction_b,
                            256, 256, 256,
                            2048, 2048, 2048,
                            0, 0, 0);
        // Compare SNAX GEMM result with golden model
        if (snrt_is_compute_core()) {
            *(local_err) = 0;
            *(local_err) += check_result(local_c, C_golden, Batch, M, N, strideInnermostC,
                                ldC, strideC);
        }
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()){
            snrt_dma_start_1d(&ERROR, local_err, sizeof(uint32_t));
        }
        snrt_cluster_hw_barrier();
    }
    // Wait for all cores done
    snrt_global_barrier();

    //return to host
    return_to_cva6(SYNC_ALL);
}