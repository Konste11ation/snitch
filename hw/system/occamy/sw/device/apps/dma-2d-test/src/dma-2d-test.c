// Copyright 2023 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
#include "data.h"

#include "snax-gemm-lib.h"
#include "snax-gemm-params.h"

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
    int8_t *local_a1;
    int32_t *local_c;
    

    // First do the 16x16x16 GeMM
    if(snrt_global_core_idx()<group0_bound_upper){
        local_err = (uint32_t*)snrt_l1_next();
        local_a = (int8_t*)(local_err+ 256);
        local_b = local_a + delta_local_a * sizeof(int8_t);
        local_c = (int32_t*)(local_b + delta_local_b * sizeof(int8_t));
        local_a1= local_a + 4096; 
        *local_err = 0;
        if(snrt_is_dm_core()){
            uint32_t dma_start = mcycle();
            snrt_dma_start_2d(local_a,A,64,4096,0,2);
            // rep = 2 means issue 2 transactions
            snrt_dma_wait_all();
            uint32_t dma_end = mcycle();
        }
    }
    // Wait for all cores done
    snrt_global_barrier();
    if(snrt_global_core_idx()<group0_bound_upper){
        if(snrt_is_compute_core()){
            uint32_t err = 0;
            for (int i = 0; i < 64; i++)
            {
                if(A[i]!=*(local_a+i)){
                    err++;
                }
            }
            for (int i = 0; i < 64; i++)
            {
                if(A[i]!=*(local_a1+i)){
                    err++;
                }
            }
            *local_err = err;           
        }
        snrt_cluster_hw_barrier();
        if(snrt_is_dm_core()){
            snrt_dma_start_1d(&ERROR,local_err,1);
        }
        
    }
    snrt_global_barrier();


    //return to host
    return_to_cva6(SYNC_ALL);
}