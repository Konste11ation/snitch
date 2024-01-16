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
            golden_idx = i * meshCol + j;
            // generate the start address of
            // sub-matrix of output C in TCDM
            // according to the strides definition
            out_addr = output + i * meshCol + j;
            // Check if output is same as golden output
            if ((int32_t)*out_addr != *(output_golden + golden_idx)) {
                err++;
            }
        }
    }

    return err;
}

uint32_t check_result_batch(int8_t Batch, int32_t* output, int32_t* output_golden) {

    uint32_t err = 0;
    uint32_t* output_batch;
    uint32_t* output_golden_batch;
    for (int b = 0; b < Batch; b++) {
        output_batch = output + b * M * N * meshRow * meshCol;
        output_golden_batch = output_golden + b * M * N * meshRow * meshCol;
        err += check_result_mod(output_batch,output_golden_batch);
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

	// Core            hard_id   Global Core idx = hartid - base_hardid = hartid - 1
	// ---------------------------
	// CVA6            0
	// ---------------------------
	// Group0:SNAX     1         0
	// Group0:SNAX     2         1
	// ---------------------------
	// Group1:SNAX     3         2
	// Group1:SNAX     4         3
	// ---------------------------
	// Group2:SNAX     5         4
	// Group2:SNAX     6         5
	// ---------------------------
	// Group3:SNAX     7         6
	// Group3:SNAX     8         7
	// ---------------------------

	uint32_t group0_bound_lower = 0;
	uint32_t group0_bound_upper = snrt_cluster_core_num();
	uint32_t group1_bound_lower = group0_bound_upper;
	uint32_t group1_bound_upper = 2*snrt_cluster_core_num();
	uint32_t group2_bound_lower = group1_bound_upper;
	uint32_t group2_bound_upper = 3*snrt_cluster_core_num();
	uint32_t group3_bound_lower = group2_bound_upper;
	uint32_t group3_bound_upper = 4*snrt_cluster_core_num();

	uint32_t size_setting = gen_size_config((uint8_t)32, (uint8_t)1, (uint8_t)1, (uint8_t)1);

	uint32_t subtraction_setting = gen_subtraction_config(subtraction_a, subtraction_b);

	// Prepare addresses in TCDM for 4 cores


	// TCDM Memory for Cluster0
	int32_t* local_err_0;
	int8_t** local_addr_A_0;
	int8_t** local_addr_B_0;
	int8_t*  local_a_0;
	int8_t*  local_b_0;
	int32_t* local_c_0;

	// TCDM Memory for Cluster1
	int32_t* local_err_1;
	int8_t*  local_a_1;
	int8_t*  local_b_1;
	int32_t* local_c_1;
	int8_t** local_a_C0_1;
	int8_t** local_b_C0_1;
	int32_t** local_c_C0_1;
	int8_t** local_a_C2_1;
	int8_t** local_b_C2_1;
	int32_t** local_c_C2_1;
	int8_t** local_a_C3_1;
	int8_t** local_b_C3_1;
	int32_t** local_c_C3_1;

	// TCDM Memory for Cluster2
	int32_t* local_err_2;
	int8_t*  local_a_2;
	int8_t*  local_b_2;
	int32_t* local_c_2;
	int8_t** local_a_C0_2;
	int8_t** local_b_C0_2;
	int32_t** local_c_C0_2;
	int8_t** local_a_C1_2;
	int8_t** local_b_C1_2;
	int32_t** local_c_C1_2;
	int8_t** local_a_C3_2;
	int8_t** local_b_C3_2;
	int32_t** local_c_C3_2;

	// TCDM Memory for Cluster3
	int32_t* local_err_3;
	int8_t*  local_a_3;
	int8_t*  local_b_3;
	int32_t* local_c_3;
	int8_t** local_a_C0_3;
	int8_t** local_b_C0_3;
	int32_t** local_c_C0_3;
	int8_t** local_a_C1_3;
	int8_t** local_b_C1_3;
	int32_t** local_c_C1_3;
	int8_t** local_a_C2_3;
	int8_t** local_b_C2_3;
	int32_t** local_c_C2_3;



	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
		// Start a new row to store err
		local_err_0 = (int32_t*)snrt_l1_next();
		local_addr_A_0 = (int8_t**)(local_err_0 + 1);
		local_addr_B_0 = (int8_t**)(local_addr_A_0 + 4);

		// Space to store matrix
		local_a_0 = (int8_t*)(local_err_0 + 256 * sizeof(int8_t));
		local_b_0 = local_a_0 + 64* sizeof(int8_t);
		local_c_0 = (int32_t*)(local_b_0 + 16384 * sizeof(int8_t));

		C0_ADDR_a = local_a_0;
		C0_ADDR_b = local_b_0;
		C0_ADDR_c = local_c_0;

		*local_err_0 = 0;
		*local_addr_A_0 = I0;
		*local_addr_B_0 = W0;
	}

	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		// Start a new row to store err
		local_err_1 = (int32_t*)snrt_l1_next();
		// Space to store matrix
		local_a_1 = (int8_t*)(local_err_1 + 256 * sizeof(int8_t));
		local_b_1 = local_a_1 + 64 * sizeof(int8_t);
		local_c_1 = (int32_t*)(local_b_1 + 16384 * sizeof(int8_t));
		local_a_C0_1 = (int8_t**)(local_err_1 + 1);
		local_b_C0_1 = (int8_t**)(local_a_C0_1 + 4);
		local_a_C2_1 = (int8_t**)(local_b_C0_1 + 4);
		local_b_C2_1 = (int8_t**)(local_a_C2_1 + 4);
		local_a_C3_1 = (int8_t**)(local_b_C2_1 + 4);
		local_b_C3_1 = (int8_t**)(local_a_C3_1 + 4);

		C1_ADDR_a = local_a_1;
		C1_ADDR_b = local_b_1;
		C1_ADDR_c = local_c_1;

		*local_err_1 = 0;
	}

	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		// Start a new row to store err
		local_err_2 = (int32_t*)snrt_l1_next();
		// Space to store matrix
		local_a_2 = (int8_t*)(local_err_2 + 256 * sizeof(int8_t));
		local_b_2 = local_a_2 + 64 * sizeof(int8_t);
		local_c_2 = (int32_t*)(local_b_2 + 16384 * sizeof(int8_t));
		local_a_C0_2 = (int8_t**)(local_err_2 + 1);
		local_b_C0_2 = (int8_t**)(local_a_C0_2 + 4);
		local_a_C1_2 = (int8_t**)(local_b_C0_2 + 4);
		local_b_C1_2 = (int8_t**)(local_a_C1_2 + 4);
		local_a_C3_2 = (int8_t**)(local_b_C1_2 + 4);
		local_b_C3_2 = (int8_t**)(local_a_C3_2 + 4);

		C2_ADDR_a = local_a_2;
		C2_ADDR_b = local_b_2;
		C2_ADDR_c = local_c_2;

		*local_err_2 = 0;
	}

	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		// Start a new row to store err
		local_err_3 = (int32_t*)snrt_l1_next();
		// Space to store matrix
		local_a_3 = (int8_t*)(local_err_3 + 256 * sizeof(int8_t));
		local_b_3 = local_a_3 + 64 * sizeof(int8_t);
		local_c_3 = (int32_t*)(local_b_3 + 16384 * sizeof(int8_t));
		local_a_C0_3 = (int8_t**)(local_err_3 + 1);
		local_b_C0_3 = (int8_t**)(local_a_C0_3 + 4);
		local_a_C1_3 = (int8_t**)(local_b_C0_3 + 4);
		local_b_C1_3 = (int8_t**)(local_a_C1_3 + 4);
		local_a_C2_3 = (int8_t**)(local_b_C1_3 + 4);
		local_b_C2_3 = (int8_t**)(local_a_C2_3 + 4);

		C3_ADDR_a = local_a_3;
		C3_ADDR_b = local_b_3;
		C3_ADDR_c = local_c_3;

		*local_err_3 = 0;
	}

	snrt_global_barrier();

	//Cluster0 already got the matrix A and B addr


	//Cluster1 need the addr of a and b for Cluster0 2 3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  *local_a_C0_1 = C0_ADDR_a;
	  *local_b_C0_1 = C0_ADDR_b;
	  *local_a_C2_1 = C2_ADDR_a;
	  *local_b_C2_1 = C2_ADDR_b;
	  *local_a_C3_1 = C3_ADDR_a;
	  *local_b_C3_1 = C3_ADDR_b;
	}

	//Cluster2 need the addr of a and b for Cluster0 1 3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  *local_a_C0_2 = C0_ADDR_a;
	  *local_b_C0_2 = C0_ADDR_b;
	  *local_a_C1_2 = C1_ADDR_a;
	  *local_b_C1_2 = C1_ADDR_b;
	  *local_a_C3_2 = C3_ADDR_a;
	  *local_b_C3_2 = C3_ADDR_b;
	}

	//Cluster3 need the addr of a and b for Cluster0 1 2
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  *local_a_C0_3 = C0_ADDR_a;
	  *local_b_C0_3 = C0_ADDR_b;
	  *local_a_C1_3 = C1_ADDR_a;
	  *local_b_C1_3 = C1_ADDR_b;
	  *local_a_C2_3 = C2_ADDR_a;
	  *local_b_C2_3 = C2_ADDR_b;
	}

	snrt_global_barrier();

//+----------+------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//|          |            | T0   | T1     | T2     | T3     | T4     | T5     | T6     | T7     | T8     | T9     | T10    | T11    |        |
//+----------+------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//| Cluster0 | DMA        | In_0 | In_1   | In_2   | In_3   | In_4   | In_5   |        |        |        |        |        |        |        |
//|          +------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//|          | Compute    |      | Out0_0 | Out0_1 | Out0_2 | Out0_3 | Out0_4 | Out0_5 |        |        |        |        |        |        |
//+----------+------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//| Cluster1 | DMA        |      |        | Out0_0 | Out0_1 | Out0_2 | Out0_3 | Out0_4 | Out0_5 |        |        |        |        |        |
//|          +------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//|          | Compute    |      |        |        | Out1_0 | Out1_1 | Out1_2 | Out1_3 | Out1_4 | Out1_5 |        |        |        |        |
//+----------+------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//| Cluster3 | DMA        |      |        |        |        | Out1_0 | Out1_1 | Out1_2 | Out1_3 | Out1_4 | Out1_5 |        |        |        |
//|          +------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//|          | Compute    |      |        |        |        |        | Out2_0 | Out2_1 | Out2_2 | Out2_3 | Out2_4 | Out2_5 |        |        |
//+----------+------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//| Cluster2 | DMA C3->C2 |      |        |        |        |        |        | Out2_0 | Out2_1 | Out2_2 | Out2_3 | Out2_4 | Out2_5 |        |
//|          +------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//|          | DMA C0->C2 |      | In_0   | In_1   | In_2   | In_3   | In_4   | In_5   |        |        |        |        |        |        |
//|          +------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
//|          | Compute    |      |        |        |        |        |        |        | Out3_0 | Out3_1 | Out3_2 | Out3_3 | Out3_4 | Out3_5 |
//+----------+------------+------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
    
	// ******************************************************************//
	//                           START THE GEMM                          //
	// ******************************************************************//


	// Cluster 0 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 0, *local_addr_A_0 + 0, 8 * 8 * sizeof(int8_t), 1024, 64, 32);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A00_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 0, *local_addr_A_0 + 0, 8 * 8 * sizeof(int8_t), 1024, 64, 32);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A00_end = mcycle();
	    }
	}



	// C2_load_A00_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C0_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 0, *local_addr_A_0 + 0, 8 * 8 * sizeof(int8_t), 1024, 64, 32);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A00_end = mcycle();
	    }
	}



	// C1_load_A00_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_1, *local_a_C0_1, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_A00_from_C0_end = mcycle();
		}
	}


	// C2_load_A00_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C0_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 0, *local_addr_A_0 + 0, 8 * 8 * sizeof(int8_t), 1024, 64, 32);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A00_end = mcycle();
	    }
	}



	// C1_load_A00_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_1, *local_a_C0_1, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_A00_from_C0_end = mcycle();
		}
	}


	// C2_load_A00_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C0_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 0, *local_addr_A_0 + 0, 8 * 8 * sizeof(int8_t), 1024, 64, 32);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A00_end = mcycle();
	    }
	}



	// C1_load_A00_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_1, *local_a_C0_1, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_A00_from_C0_end = mcycle();
		}
	}


	// C3_load_A00_from_C1
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_A00_from_C1_start = mcycle();
			snrt_dma_start_2d(local_a_3, *local_a_C1_3, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_A00_from_C1_end = mcycle();
		}
	}


	// C2_load_A00_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C0_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 0, *local_addr_A_0 + 0, 8 * 8 * sizeof(int8_t), 1024, 64, 32);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A00_end = mcycle();
	    }
	}



	// C1_load_A00_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_1, *local_a_C0_1, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_A00_from_C0_end = mcycle();
		}
	}


	// C3_load_A00_from_C1
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_A00_from_C1_start = mcycle();
			snrt_dma_start_2d(local_a_3, *local_a_C1_3, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_A00_from_C1_end = mcycle();
		}
	}


	// C2_load_A00_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C0_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// C1_load_A00_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_1, *local_a_C0_1, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_A00_from_C0_end = mcycle();
		}
	}


	// C3_load_A00_from_C1
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_A00_from_C1_start = mcycle();
			snrt_dma_start_2d(local_a_3, *local_a_C1_3, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_A00_from_C1_end = mcycle();
		}
	}


	// C2_load_A00_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C0_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C0_end = mcycle();
		}
	}


	// C2_load_A00_from_C3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C3_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C3_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C3_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// C1_load_A00_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_1, *local_a_C0_1, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_A00_from_C0_end = mcycle();
		}
	}


	// C3_load_A00_from_C1
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_A00_from_C1_start = mcycle();
			snrt_dma_start_2d(local_a_3, *local_a_C1_3, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_A00_from_C1_end = mcycle();
		}
	}


	// C2_load_A00_from_C3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C3_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C3_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C3_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// C3_load_A00_from_C1
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_A00_from_C1_start = mcycle();
			snrt_dma_start_2d(local_a_3, *local_a_C1_3, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_A00_from_C1_end = mcycle();
		}
	}


	// C2_load_A00_from_C3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C3_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C3_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C3_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// C3_load_A00_from_C1
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_A00_from_C1_start = mcycle();
			snrt_dma_start_2d(local_a_3, *local_a_C1_3, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_A00_from_C1_end = mcycle();
		}
	}


	// C2_load_A00_from_C3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C3_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C3_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C3_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// C2_load_A00_from_C3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C3_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C3_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C3_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// C2_load_A00_from_C3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A00_from_C3_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C3_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 32);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A00_from_C3_end = mcycle();
		}
	}

	// ******************************************************************//
	//                           END THE GEMM                            //
	// ******************************************************************//

	//return to host
	return_to_cva6(SYNC_ALL);

}

