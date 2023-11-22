// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include "snrt.h"

#include "data.h"

void snax_compute(uint32_t *a,uint32_t *b,uint32_t *c,uint32_t *o){
    uint32_t csr_set = mcycle();
            // Set addresses
    write_csr(0x3d0, (uint32_t)a);
    write_csr(0x3d1, (uint32_t)b);
    write_csr(0x3d2, (uint32_t)c);
    write_csr(0x3d3, (uint32_t)o);
    // Set configs
    write_csr(0x3d4, 1);   // Number of iterations
    write_csr(0x3d5, 19);  // Vector length

    // Write start CSR to launch accelerator
    write_csr(0x3c0, 0);

    // Start of CSR start and poll until accelerator finishes
    uint32_t mac_start = mcycle();

    uint32_t break_poll;

    while (1) {
        // 0x3c3 is the CSR address for accelerator status
        break_poll = read_csr(0x3c3);
        if (break_poll == 0) {
            break;
        };
    };

    uint32_t mac_end = mcycle();
}



int main() {
    
    // Core            hard_id   Global Core idx = hartid - base_hardid = hartid - 1
    // CVA6            0
    // Group0:SNAX     1         0
    // Group0:DM Core  2         1
    // Group1:SNAX     3         2
    // Group1:DM Core  4         3

    // First assign the job at the Group 0
    uint32_t group0_bound_lower = 0;
    uint32_t group0_bound_upper = snrt_cluster_core_num(); // 2
    uint32_t group1_bound_lower = group0_bound_upper;
    uint32_t group1_bound_upper = 2*snrt_cluster_core_num(); // 4
    uint32_t *group0_a, *group0_b, *group0_c, *group0_o;
    uint32_t *group1_a, *group1_b, *group1_c, *group1_o;
    if(snrt_global_core_idx()<group0_bound_upper){
        

        // Allocate space in TCDM
        group0_a = (uint32_t *)snrt_l1_next();
        group0_b = group0_a + VEC_LEN;
        group0_c = group0_b + VEC_LEN;
        group0_o = group0_c + 1;

        uint32_t dma_pre_load = mcycle();

        // Use data mover core to bring data from L3 to TCDM
        if (snrt_is_dm_core()) {
            size_t vector_size = VEC_LEN * sizeof(uint32_t);
            size_t scale_size = 1 * sizeof(uint32_t);
            snrt_dma_start_1d(group0_a, A, vector_size);
            snrt_dma_start_1d(group0_b, B, vector_size);
            snrt_dma_start_1d(group0_c, &C, scale_size);
        }

        // Wait until DMA transfer is done
        snrt_cluster_hw_barrier();

        // Read the mcycle CSR (this is our way to mark/delimit a specific
        // code region for benchmarking)
        uint32_t pre_is_compute_core = mcycle();

        if (snrt_is_compute_core()) {
            snax_compute(group0_a,group0_b,group0_c,group0_o);
        }
    }
    snrt_global_barrier();
    // Now we are at Group 1
    if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
        
        // Allocate space in TCDM
        group1_a = (uint32_t *)snrt_l1_next();
        group1_b = group1_a + VEC_LEN;
        group1_c = group1_b + VEC_LEN;
        group1_o = group1_c + 1;
        uint32_t dma_pre_load = mcycle();

        // Use data mover core to bring data from L3 to TCDM
        if (snrt_is_dm_core()) {
            size_t vector_size = VEC_LEN * sizeof(uint32_t);
            size_t scale_size = 1 * sizeof(uint32_t);
            snrt_dma_start_1d(group1_a, group0_a, vector_size);
            snrt_dma_start_1d(group1_b, group0_b, vector_size);
            snrt_dma_start_1d(group1_c, group0_c, scale_size);
        }
        // Wait until DMA transfer is done
        snrt_cluster_hw_barrier();

        // Read the mcycle CSR (this is our way to mark/delimit a specific
        // code region for benchmarking)
        uint32_t pre_is_compute_core = mcycle();

        if (snrt_is_compute_core()) {
            snax_compute(group1_a,group1_b,group1_c,group1_o);
        }
    }


    return_to_cva6(SYNC_ALL);
}
