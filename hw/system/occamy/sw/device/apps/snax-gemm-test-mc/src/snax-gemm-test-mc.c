// Copyright 2023 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
#include "data.h"

#include "snax-gemm-lib.h"
#include "snax-gemm-params.h"



uint32_t check_result_mod(int32_t* output, int32_t* output_golden) {
    /*
     * Compare output to output_golden with length
     */
    uint32_t err = 0;
    uint32_t golden_idx;
    int32_t* out_addr;

 
    for (int i = 0; i < meshRow; i++) {
        for (int j = 0; j < meshCol; j++) {
            // element index of golden results
            golden_idx =
                i * meshCol + j ;
            // generate the start address of
            // sub-matrix of output C in TCDM
            // according to the strides definition
            out_addr =
                output + i * meshCol + j;
            // Check if output is same as golden output
            if ((int32_t)*out_addr != *(output_golden+golden_idx)) {
                err++;
            };
        };
    }


    return err;
}

uint8_t*  C0_ADDR_a;
uint8_t*  C0_ADDR_b;
uint32_t* C0_ADDR_c;

uint8_t*  C1_ADDR_a;
uint8_t*  C1_ADDR_b;
uint32_t* C1_ADDR_c;

uint8_t*  C2_ADDR_a;
uint8_t*  C2_ADDR_b;
uint32_t* C2_ADDR_c;

uint8_t*  C3_ADDR_a;
uint8_t*  C3_ADDR_b;
uint32_t* C3_ADDR_c;

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

    // Prepare addresses in TCDM for 4 cores

    int32_t* local_err_0;
    int8_t* local_a_0;
    int8_t* local_b_0;
    int32_t* local_c_0;

    int32_t* local_err_1;
    int8_t* local_a_1;
    int8_t* local_b_1;
    int32_t* local_c_1;

    int32_t* local_err_2;
    int8_t* local_a_2;
    int8_t* local_b_2;
    int32_t* local_c_2;
  
    int32_t* local_err_3;
    int8_t* local_a_3;
    int8_t* local_b_3;
    int32_t* local_c_3;
    
    uint32_t size_setting = gen_size_config(Batch, 
                                           1, 2, 1);

    uint32_t subtraction_setting = gen_subtraction_config(subtraction_a, subtraction_b);
                                          
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C0 | DMA     | A00 | A01 | B00 | B10 | A10 | A11 | B01 | B11 |     |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     | C11 |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C1 | DMA     |     | A00 | A01 |     |     |     |     | B01 | B11 |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |     |     | C12 |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C2 | DMA     |     |     |     | B00 | B10 | A10 | A11 |     |     |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     | C21 |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C3 | DMA     |     |     |     |     |     |     | A10 | A11 | B01 | B11 |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |     |     |     | C22 |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    
    // First initilize the GeMM Config and tcdm address in all cores
    if(snrt_global_core_idx()<group0_bound_upper){

        // Start a new row to store err
        local_err_0 = (int32_t*)snrt_l1_next();
        // Space to store matrix
        local_a_0 = (int8_t*)(local_err_0 + 256 * sizeof(int8_t)); // across a row in TCDM
        local_b_0 = local_a_0 + delta_local_a * sizeof(int8_t);
        local_c_0 = (int32_t*)(local_b_0 + delta_local_b * sizeof(int8_t));

        C0_ADDR_a = local_a_0;
        C0_ADDR_b = local_b_0;
        C0_ADDR_c = local_c_0;

        *local_err_0 = 0;        
    }

    if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){

        // Start a new row to store err
        local_err_1 = (int32_t*)snrt_l1_next();
        // Space to store matrix
        local_a_1 = (int8_t*)(local_err_1 + 256 * sizeof(int8_t)); // across a row in TCDM
        local_b_1 = local_a_1 + delta_local_a * sizeof(int8_t);
        local_c_1 = (int32_t*)(local_b_1 + delta_local_b * sizeof(int8_t));

        C1_ADDR_a = local_a_1;
        C1_ADDR_b = local_b_1;
        C1_ADDR_c = local_c_1;

        *local_err_1 = 0;  
    }

    if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
        // Start a new row to store err
        local_err_2 = (int32_t*)snrt_l1_next();
        // Space to store matrix
        local_a_2 = (int8_t*)(local_err_2 + 256 * sizeof(int8_t)); // across a row in TCDM
        local_b_2 = local_a_2 + delta_local_a * sizeof(int8_t);
        local_c_2 = (int32_t*)(local_b_2 + delta_local_b * sizeof(int8_t));
        
        C2_ADDR_a = local_a_2;
        C2_ADDR_b = local_b_2;
        C2_ADDR_c = local_c_2;

        if (snrt_is_compute_core()){
            // Set GEMM configuration CSR
            set_batch_gemm(size_setting, 
                           local_a_2, local_b_2, 
                           subtraction_setting,
                           local_c_2, 
                           strideInnermostA, 
                           strideInnermostB,
                           strideInnermostC, 
                           ldA, 
                           ldB, 
                           ldC, 
                           strideA, 
                           strideB,
                           strideC);
        }
        snrt_cluster_hw_barrier();

        *local_err_2 = 0;  
    }

    if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
        // Start a new row to store err
        local_err_3 = (int32_t*)snrt_l1_next();
        // Space to store matrix
        local_a_3 = (int8_t*)(local_err_3 + 256 * sizeof(int8_t)); // across a row in TCDM
        local_b_3 = local_a_3 + delta_local_a * sizeof(int8_t);
        local_c_3 = (int32_t*)(local_b_3 + delta_local_b * sizeof(int8_t));

        C3_ADDR_a = local_a_3;
        C3_ADDR_b = local_b_3;
        C3_ADDR_c = local_c_3;

        *local_err_3 = 0;  
    }
    // wait for the setting to be done
    snrt_global_barrier();

    // ******************************************************************//
    //                           START THE GEMM                          //
    // ******************************************************************//
    // C0 : Load the A00 from the TCDM
    //-------------------
    //                  |
    //                  v
    //+----+---------+-----+-
    //| C0 | DMA     | A00 | 
    //|    +---------+-----+-
    //|    | Compute |     | 
    //+----+---------+-----+-
    //| C1 | DMA     |     | 
    //|    +---------+-----+-
    //|    | Compute |     | 
    //+----+---------+-----+-
    //| C2 | DMA     |     | 
    //|    +---------+-----+-
    //|    | Compute |     | 
    //+----+---------+-----+-
    //| C3 | DMA     |     | 
    //|    +---------+-----+-
    //|    | Compute |     | 
    //+----+---------+-----+-
    
    if(snrt_global_core_idx()<group0_bound_upper){

        if (snrt_is_dm_core()) {
            uint32_t dma_C0_load_A00_start = mcycle();
            load_input_A_block(0,
                               0, 0,
                               1, 2,
                               local_a_0,
                               A,
                               strideInnermostA,
                               ldA,
                               strideA);
            uint32_t dma_C0_load_A00_end = mcycle();
            snrt_dma_wait_all();
        }
        
        
    }
    
    snrt_global_barrier();

    // ******************************************************************//

    // C0 : Load the A01 from the TCDM
    // C1 : Load the A00 from C0
    //-------------------------
    //                        |
    //                        v
    //+----+---------+-----+-----+
    //| C0 | DMA     | A00 | A01 |
    //|    +---------+-----+-----+
    //|    | Compute |     |     |
    //+----+---------+-----+-----+
    //| C1 | DMA     |     | A00 |
    //|    +---------+-----+-----+
    //|    | Compute |     |     |
    //+----+---------+-----+-----+
    //| C2 | DMA     |     |     |
    //|    +---------+-----+-----+
    //|    | Compute |     |     |
    //+----+---------+-----+-----+
    //| C3 | DMA     |     |     |
    //|    +---------+-----+-----+
    //|    | Compute |     |     |
    //+----+---------+-----+-----+
    
    if(snrt_global_core_idx()<group0_bound_upper){
        
        if (snrt_is_dm_core()) {
            uint32_t dma_C0_load_A01_start = mcycle();
            load_input_A_block(0,
                               0, 1,
                               1, 2,
                               local_a_0,
                               A,
                               strideInnermostA,
                               ldA,
                               strideA);
            uint32_t dma_C0_load_A01_end = mcycle();
            snrt_dma_wait_all();
        }
        
        
    }
    

    if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
              
        if(snrt_is_dm_core()){
            uint32_t dma_C1_load_A00_start = mcycle();

            snrt_dma_start_1d(C1_ADDR_a, C0_ADDR_a, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C1_load_A00_end = mcycle();
            snrt_dma_wait_all();
        }
        
        
    }
    snrt_global_barrier();

    // ******************************************************************//

    // C0 : Load the B00 from the TCDM
    // C1 : Load the A01 from C0
    //-------------------------------
    //                              |
    //                              v
    //+----+---------+-----+-----+-----+-
    //| C0 | DMA     | A00 | A01 | B00 | 
    //|    +---------+-----+-----+-----+-
    //|    | Compute |     |     |     | 
    //+----+---------+-----+-----+-----+-
    //| C1 | DMA     |     | A00 | A01 | 
    //|    +---------+-----+-----+-----+-
    //|    | Compute |     |     |     | 
    //+----+---------+-----+-----+-----+-
    //| C2 | DMA     |     |     |     | 
    //|    +---------+-----+-----+-----+-
    //|    | Compute |     |     |     | 
    //+----+---------+-----+-----+-----+-
    //| C3 | DMA     |     |     |     | 
    //|    +---------+-----+-----+-----+-
    //|    | Compute |     |     |     | 
    //+----+---------+-----+-----+-----+-
    if(snrt_global_core_idx()<group0_bound_upper){
        
        if (snrt_is_dm_core()) {
            uint32_t dma_C0_load_B00_start = mcycle();
            load_input_B_block(0,
                               0, 0,
                               1, 2,
                               local_b_0,
                               B,
                               strideInnermostB,
                               ldB,
                               strideB);
            uint32_t dma_C0_load_B00_end = mcycle();
            snrt_dma_wait_all();
        }
        
        
    }

    if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C1_load_A01_start = mcycle();
            snrt_dma_start_1d(C1_ADDR_a + 1 * strideInnermostA, C0_ADDR_a + 1 * strideInnermostA, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C1_load_A01_end = mcycle();
            snrt_dma_wait_all();
        }
        
        
    }
    snrt_global_barrier();

    // ******************************************************************//

    // C0 : Load the B10 from the TCDM
    // C2 : Load the B00 from C0
    //-------------------------------------
    //                                    |
    //                                    v
    //+----+---------+-----+-----+-----+-----+
    //| C0 | DMA     | A00 | A01 | B00 | B10 |
    //|    +---------+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+
    //| C1 | DMA     |     | A00 | A01 |     |
    //|    +---------+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+
    //| C2 | DMA     |     |     |     | B00 |
    //|    +---------+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+
    //| C3 | DMA     |     |     |     |     |
    //|    +---------+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+
    if(snrt_global_core_idx()<group0_bound_upper){
        
        if (snrt_is_dm_core()) {
            uint32_t dma_C0_load_B10_start = mcycle();
            load_input_B_block(0,
                               0, 1,
                               1, 2,
                               local_b_0,
                               B,
                               strideInnermostB,
                               ldB,
                               strideB);
            uint32_t dma_C0_load_B10_end = mcycle();
            snrt_dma_wait_all();
        }
        
        
    }
    if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C2_load_B00_start = mcycle();
            snrt_dma_start_1d(C2_ADDR_b, C0_ADDR_b, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C2_load_B00_end = mcycle();
            snrt_dma_wait_all();
        }
        
        
    }
    snrt_global_barrier();

    // ******************************************************************//

    // C0 : Load the A10 from the TCDM
    // C0 : Compute the C11
    // C2 : Load the B10 from C0
    //-------------------------------------------
    //                                          |
    //                                          v
    //+----+---------+-----+-----+-----+-----+-----+-
    //| C0 | DMA     | A00 | A01 | B00 | B10 | A10 | 
    //|    +---------+-----+-----+-----+-----+-----+-
    //|    | Compute |     |     |     |     | C11 | 
    //+----+---------+-----+-----+-----+-----+-----+-
    //| C1 | DMA     |     | A00 | A01 |     |     | 
    //|    +---------+-----+-----+-----+-----+-----+-
    //|    | Compute |     |     |     |     |     | 
    //+----+---------+-----+-----+-----+-----+-----+-
    //| C2 | DMA     |     |     |     | B00 | B10 | 
    //|    +---------+-----+-----+-----+-----+-----+-
    //|    | Compute |     |     |     |     |     | 
    //+----+---------+-----+-----+-----+-----+-----+-
    //| C3 | DMA     |     |     |     |     |     | 
    //|    +---------+-----+-----+-----+-----+-----+-
    //|    | Compute |     |     |     |     |     | 
    //+----+---------+-----+-----+-----+-----+-----+-

    if(snrt_global_core_idx()<group0_bound_upper){
        if (snrt_is_dm_core()) {
            uint32_t dma_C0_load_A10_start = mcycle();
            load_input_A_block(0,
                               1, 0,
                               1, 2,
                               local_a_0,
                               A,
                               strideInnermostA,
                               ldA,
                               strideA);
            snrt_dma_wait_all();
            uint32_t dma_C0_load_A10_end = mcycle();
        }
        
        snrt_cluster_hw_barrier();
        
        if (snrt_is_compute_core()){
            uint32_t gemm_C11_start = mcycle();


            // Set GEMM configuration CSR
            set_batch_gemm(size_setting, 
                           local_a_0, local_b_0, 
                           subtraction_setting,
                           local_c_0, 
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

            uint32_t gemm_C11_end = mcycle();
    
        }
    }

    if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C2_load_B10_start = mcycle();
            snrt_dma_start_1d(C2_ADDR_b + 1 * strideInnermostB, C0_ADDR_b + 1 * strideInnermostB, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C2_load_B10_end = mcycle();
            snrt_dma_wait_all();
        }
        
        
    } 

    snrt_global_barrier();
    // ******************************************************************//

    // C0 : Load the A11 from the TCDM
    // C2 : Load the A10 from C0    
    //-------------------------------------------------
    //                                                |
    //                                                v    
    //+----+---------+-----+-----+-----+-----+-----+-----+
    //| C0 | DMA     | A00 | A01 | B00 | B10 | A10 | A11 |
    //|    +---------+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     | C11 |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+
    //| C1 | DMA     |     | A00 | A01 |     |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+
    //| C2 | DMA     |     |     |     | B00 | B10 | A10 |
    //|    +---------+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+
    //| C3 | DMA     |     |     |     |     |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+
    if(snrt_global_core_idx()<group0_bound_upper){
        
        if (snrt_is_dm_core()) {
            uint32_t dma_C0_load_A11_start = mcycle();
            load_input_A_block(0,
                               1, 1,
                               1, 2,
                               local_a_0,
                               A,
                               strideInnermostA,
                               ldA,
                               strideA);
            uint32_t dma_C0_load_A11_end = mcycle();
            snrt_dma_wait_all();
        }
        
    }

    if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C2_load_A10_start = mcycle();
            snrt_dma_start_1d(C2_ADDR_a , C0_ADDR_a + 1 * ldA, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C2_load_A10_end = mcycle();
            snrt_dma_wait_all();
        }
        
        
    } 

    snrt_global_barrier();
    // ******************************************************************//

    // C0 : Load the B01 from the TCDM
    // C2 : Load the A11 from C0
    // C3 : Load the A10 from C2    
    //-------------------------------------------------------
    //                                                      |
    //                                                      v 
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+
    //| C0 | DMA     | A00 | A01 | B00 | B10 | A10 | A11 | B01 |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     | C11 |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+
    //| C1 | DMA     |     | A00 | A01 |     |     |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+
    //| C2 | DMA     |     |     |     | B00 | B10 | A10 | A11 |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+
    //| C3 | DMA     |     |     |     |     |     |     | A10 |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+
    if(snrt_global_core_idx()<group0_bound_upper){
        
        if (snrt_is_dm_core()) {
            uint32_t dma_C0_load_B01_start = mcycle();
            load_input_B_block(0,
                               1, 0,
                               1, 2,
                               local_b_0,
                               B,
                               strideInnermostB,
                               ldB,
                               strideB);
            snrt_dma_wait_all();
            uint32_t dma_C0_load_B01_end = mcycle();
        }
        
        
    }

    if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C2_load_A11_start = mcycle();
            snrt_dma_start_1d(C2_ADDR_a + 1 * strideInnermostA, C0_ADDR_a + 1 * ldA + 1 * strideInnermostA, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C2_load_A11_end = mcycle();
            snrt_dma_wait_all();
        }
        
    } 

    if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C3_load_A10_start = mcycle();
            snrt_dma_start_1d(C3_ADDR_a, C2_ADDR_a, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C3_load_A10_end = mcycle();
            snrt_dma_wait_all();
        }
        
    }

    snrt_global_barrier();
    // ******************************************************************//

    // C0 : Load the B11 from the TCDM
    // C1 : Load the B01 from C0
    // C2 : Compute the C21
    // C3 : Load A11 from C2    
    //-------------------------------------------------------------
    //                                                            |
    //                                                            v     
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C0 | DMA     | A00 | A01 | B00 | B10 | A10 | A11 | B01 | B11 |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     | C11 |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C1 | DMA     |     | A00 | A01 |     |     |     |     | B01 |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C2 | DMA     |     |     |     | B00 | B10 | A10 | A11 |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     | C21 |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C3 | DMA     |     |     |     |     |     |     | A10 | A11 |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+

    if(snrt_global_core_idx()<group0_bound_upper){
        if (snrt_is_dm_core()) {
            uint32_t dma_C0_load_B11_start = mcycle();
            load_input_B_block(0,
                               1, 1,
                               1, 2,
                               local_b_0,
                               B,
                               strideInnermostB,
                               ldB,
                               strideB);
            uint32_t dma_C0_load_B11_end = mcycle();
            snrt_dma_wait_all();
        }
    }

    if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C1_load_B01_start = mcycle();
            snrt_dma_start_1d(C1_ADDR_b , C0_ADDR_b + 1 * ldB, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C1_load_B01_end = mcycle();
            snrt_dma_wait_all();
        }
    }     

    if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
        if (snrt_is_compute_core()){
            uint32_t gemm_C21_start = mcycle();

            // Set GEMM configuration CSR
            set_batch_gemm(size_setting, 
                           local_a_2, local_b_2, 
                           subtraction_setting,
                           local_c_2, 
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

            uint32_t gemm_C21_end = mcycle();     
        }       
    } 

    if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C3_load_A11_start = mcycle();
            snrt_dma_start_1d(C3_ADDR_a + 1 * strideInnermostA, C2_ADDR_a + 1 * strideInnermostA, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C3_load_A11_end = mcycle();
            snrt_dma_wait_all();
        }
        
    }

    snrt_global_barrier();
    // ******************************************************************//

    // C1 : Load the B11 from C0
    // C3 : Load B01 from C1    
    //------------------------------------------------------------------
    //                                                                 |
    //                                                                 v    
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C0 | DMA     | A00 | A01 | B00 | B10 | A10 | A11 | B01 | B11 |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     | C11 |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C1 | DMA     |     | A00 | A01 |     |     |     |     | B01 | B11 |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C2 | DMA     |     |     |     | B00 | B10 | A10 | A11 |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     | C21 |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C3 | DMA     |     |     |     |     |     |     | A10 | A11 | B01 |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C1_load_B11_start = mcycle();
            snrt_dma_start_1d(C1_ADDR_b + 1 * strideInnermostB, C0_ADDR_b + 1 * ldB + 1 * strideInnermostB, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C1_load_B11_end = mcycle();
        }
        
    }

    if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C1_load_B01_start = mcycle();
            snrt_dma_start_1d(C3_ADDR_b, C1_ADDR_b, 8 * 8 * sizeof(int8_t));
            uint32_t dma_C1_load_B01_end = mcycle();
        }
        
    }   

    snrt_global_barrier();
    // ******************************************************************//

    // C1 : Compute C12
    // C3 : Load B11 from C1    
    //------------------------------------------------------------------------
    //                                                                        |
    //                                                                        v     
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-
    //| C0 | DMA     | A00 | A01 | B00 | B10 | A10 | A11 | B01 | B11 |     |     | 
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-
    //|    | Compute |     |     |     |     | C11 |     |     |     |     |     | 
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-
    //| C1 | DMA     |     | A00 | A01 |     |     |     |     | B01 | B11 |     | 
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-
    //|    | Compute |     |     |     |     |     |     |     |     |     | C12 | 
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-
    //| C2 | DMA     |     |     |     | B00 | B10 | A10 | A11 |     |     |     | 
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-
    //|    | Compute |     |     |     |     |     |     |     | C21 |     |     | 
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-
    //| C3 | DMA     |     |     |     |     |     |     | A10 | A11 | B01 | B11 | 
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-
    //|    | Compute |     |     |     |     |     |     |     |     |     |     | 
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-
    if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
        if (snrt_is_compute_core()){
            uint32_t gemm_C12_start = mcycle();

            set_batch_gemm(size_setting, 
                           local_a_1, local_b_1, 
                           subtraction_setting,
                           local_c_1, 
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

            uint32_t gemm_C12_end = mcycle();     
        }        
        
    }

    if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
        
        if(snrt_is_dm_core()){
            uint32_t dma_C1_load_B11_start = mcycle();
            snrt_dma_start_1d(C3_ADDR_b + 1 * strideInnermostB, C1_ADDR_b + 1 * strideInnermostB, meshRow * tileSize * sizeof(int8_t));
            uint32_t dma_C1_load_B11_end = mcycle();
        }
        
    } 

    snrt_global_barrier();
    // // ******************************************************************//

    // C3 : Compute C22    
    //-------------------------------------------------------------------------------
    //                                                                              |
    //                                                                              v     
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C0 | DMA     | A00 | A01 | B00 | B10 | A10 | A11 | B01 | B11 |     |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     | C11 |     |     |     |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C1 | DMA     |     | A00 | A01 |     |     |     |     | B01 | B11 |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |     |     | C12 |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C2 | DMA     |     |     |     | B00 | B10 | A10 | A11 |     |     |     |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     | C21 |     |     |     |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //| C3 | DMA     |     |     |     |     |     |     | A10 | A11 | B01 | B11 |     |
    //|    +---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    | Compute |     |     |     |     |     |     |     |     |     |     | C22 |
    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

    if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
        if (snrt_is_compute_core()){
            uint32_t gemm_C22_start = mcycle();

            // Set GEMM configuration CSR
                if (snrt_is_compute_core()){
                    // Set GEMM configuration CSR
                    set_batch_gemm(size_setting, 
                                local_a_3, local_b_3, 
                                subtraction_setting,
                                local_c_3, 
                                strideInnermostA, 
                                strideInnermostB,
                                strideInnermostC, 
                                ldA, 
                                ldB, 
                                ldC, 
                                strideA, 
                                strideB,
                                strideC);
                }
            start_batch_gemm();
            wait_batch_gemm();

            uint32_t gemm_C22_end = mcycle();     
        }        
        
    }

    snrt_global_barrier();

    // ******************************************************************//
    //                           END THE GEMM                            //
    // ******************************************************************//

    // Check Results
    // Compare SNAX GEMM result with golden model

    if(snrt_global_core_idx()<group0_bound_upper){

        if (snrt_is_compute_core()) {
            
            *local_err_0 += check_result_mod(C0_ADDR_c, C_golden);
        }        
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()){
            snrt_dma_start_1d(ERROR, local_err_0, sizeof(uint32_t));
        }
    }

    if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){

        if (snrt_is_compute_core()) {
            
            *local_err_1 += check_result_mod(C1_ADDR_c, C_golden + 64);
        }        
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()){
            snrt_dma_start_1d(ERROR + 1, local_err_1, sizeof(uint32_t));
        }
    }

    if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){

        if (snrt_is_compute_core()) {
            
            *local_err_2 += check_result_mod(C2_ADDR_c, C_golden + 128);
        }        
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()){
            snrt_dma_start_1d(ERROR + 2, local_err_2, sizeof(uint32_t));
        }
    }

    if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){

        if (snrt_is_compute_core()) {
            
            *local_err_3 += check_result_mod(C3_ADDR_c, C_golden + 192);
        }        
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()){
            snrt_dma_start_1d(ERROR + 3, local_err_3, sizeof(uint32_t));
        }
    }
    snrt_global_barrier();
    //return to host
    return_to_cva6(SYNC_ALL);
}