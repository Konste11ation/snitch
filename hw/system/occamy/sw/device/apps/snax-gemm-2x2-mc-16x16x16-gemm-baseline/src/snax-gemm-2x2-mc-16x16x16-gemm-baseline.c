#include "data.h"
#include "snax-gemm-lib.h"
#include "snax-gemm-params.h"


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
	int8_t** local_addr_A_1;
	int8_t** local_addr_B_1;
	int8_t*  local_a_1;
	int8_t*  local_b_1;
	int32_t* local_c_1;


	// TCDM Memory for Cluster2
	int32_t* local_err_2;
	int8_t** local_addr_A_2;
	int8_t** local_addr_B_2;
	int8_t*  local_a_2;
	int8_t*  local_b_2;
	int32_t* local_c_2;


	// TCDM Memory for Cluster3
	int32_t* local_err_3;
	int8_t** local_addr_A_3;
	int8_t** local_addr_B_3;
	int8_t*  local_a_3;
	int8_t*  local_b_3;
	int32_t* local_c_3;


	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
		// Start a new row to store err
		local_err_0 = (int32_t*)snrt_l1_next();
		local_addr_A_0 = (int8_t**)(local_err_0 + 1);
		local_addr_B_0 = (int8_t**)(local_addr_A_0 + 4);

		// Space to store matrix
		local_a_0 = (int8_t*)(local_err_0 + 256 * sizeof(int8_t));
		local_b_0 = local_a_0 + 64* sizeof(int8_t);
		local_c_0 = (int32_t*)(local_b_0 + 16384 * sizeof(int8_t));

		*local_err_0 = 0;
		*local_addr_A_0 = A;
		*local_addr_B_0 = B;
	}


	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		// Start a new row to store err
		local_err_1 = (int32_t*)snrt_l1_next();
		local_addr_A_1 = (int8_t**)(local_err_1 + 1);
		local_addr_B_1 = (int8_t**)(local_addr_A_1 + 4);

		// Space to store matrix
		local_a_1 = (int8_t*)(local_err_1 + 256 * sizeof(int8_t));
		local_b_1 = local_a_1 + 64* sizeof(int8_t);
		local_c_1 = (int32_t*)(local_b_1 + 16384 * sizeof(int8_t));

		*local_err_1 = 0;
		*local_addr_A_1 = A;
		*local_addr_B_1 = B;
	}


	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		// Start a new row to store err
		local_err_2 = (int32_t*)snrt_l1_next();
		local_addr_A_2 = (int8_t**)(local_err_2 + 1);
		local_addr_B_2 = (int8_t**)(local_addr_A_2 + 4);

		// Space to store matrix
		local_a_2 = (int8_t*)(local_err_2 + 256 * sizeof(int8_t));
		local_b_2 = local_a_2 + 64* sizeof(int8_t);
		local_c_2 = (int32_t*)(local_b_2 + 16384 * sizeof(int8_t));

		*local_err_2 = 0;
		*local_addr_A_2 = A;
		*local_addr_B_2 = B;
	}


	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		// Start a new row to store err
		local_err_3 = (int32_t*)snrt_l1_next();
		local_addr_A_3 = (int8_t**)(local_err_3 + 1);
		local_addr_B_3 = (int8_t**)(local_addr_A_3 + 4);

		// Space to store matrix
		local_a_3 = (int8_t*)(local_err_3 + 256 * sizeof(int8_t));
		local_b_3 = local_a_3 + 64* sizeof(int8_t);
		local_c_3 = (int32_t*)(local_b_3 + 16384 * sizeof(int8_t));

		*local_err_3 = 0;
		*local_addr_A_3 = A;
		*local_addr_B_3 = B;
	}


	snrt_global_barrier();

	snrt_global_barrier();

//+----+-----+-----+-----+-----+
//|    | T0  | T1  | T2  | T3  |
//+----+-----+-----+-----+-----+
//| C0 | A00 | A01 | B00 | B10 |
//+----+-----+-----+-----+-----+
//| C1 | A00 | A01 | B01 | B11 |
//+----+-----+-----+-----+-----+
//| C2 | A10 | A11 | B00 | B10 |
//+----+-----+-----+-----+-----+
//| C3 | A10 | A11 | B01 | B11 |
//+----+-----+-----+-----+-----+
    
	// ******************************************************************//
	//                           START THE GEMM                          //
	// ******************************************************************//


	// Cluster 0 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 0, *local_addr_A_0 + 0, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A00_end = mcycle();
	    }
	}



	// Cluster 1 Load the A00 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_1 + 0, *local_addr_A_1 + 0, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_A00_end = mcycle();
	    }
	}



	// Cluster 2 Load the A10 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_A10_start = mcycle();
	    snrt_dma_start_2d(local_a_2 + 512, *local_addr_A_2 + 128, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_A10_end = mcycle();
	    }
	}



	// Cluster 3 Load the A10 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_A10_start = mcycle();
	    snrt_dma_start_2d(local_a_3 + 512, *local_addr_A_3 + 128, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_A10_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A01 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A01_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 256, *local_addr_A_0 + 64, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A01_end = mcycle();
	    }
	}



	// Cluster 1 Load the A01 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_A01_start = mcycle();
	    snrt_dma_start_2d(local_a_1 + 256, *local_addr_A_1 + 64, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_A01_end = mcycle();
	    }
	}



	// Cluster 2 Load the A11 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_A11_start = mcycle();
	    snrt_dma_start_2d(local_a_2 + 768, *local_addr_A_2 + 192, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_A11_end = mcycle();
	    }
	}



	// Cluster 3 Load the A11 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_A11_start = mcycle();
	    snrt_dma_start_2d(local_a_3 + 768, *local_addr_A_3 + 192, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_A11_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_B00_start = mcycle();
	    snrt_dma_start_2d(local_b_0 + 0, *local_addr_B_0 + 0, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_B00_end = mcycle();
	    }
	}



	// Cluster 1 Load the B01 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_B01_start = mcycle();
	    snrt_dma_start_2d(local_b_1 + 512, *local_addr_B_1 + 128, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_B01_end = mcycle();
	    }
	}



	// Cluster 2 Load the B00 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_B00_start = mcycle();
	    snrt_dma_start_2d(local_b_2 + 0, *local_addr_B_2 + 0, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_B00_end = mcycle();
	    }
	}



	// Cluster 3 Load the B01 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_B01_start = mcycle();
	    snrt_dma_start_2d(local_b_3 + 512, *local_addr_B_3 + 128, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_B01_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B10 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_B10_start = mcycle();
	    snrt_dma_start_2d(local_b_0 + 256, *local_addr_B_0 + 64, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_B10_end = mcycle();
	    }
	}



	// Cluster 1 Load the B11 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_B11_start = mcycle();
	    snrt_dma_start_2d(local_b_1 + 768, *local_addr_B_1 + 192, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_B11_end = mcycle();
	    }
	}



	// Cluster 2 Load the B10 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_B10_start = mcycle();
	    snrt_dma_start_2d(local_b_2 + 256, *local_addr_B_2 + 64, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_B10_end = mcycle();
	    }
	}



	// Cluster 3 Load the B11 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_B11_start = mcycle();
	    snrt_dma_start_2d(local_b_3 + 768, *local_addr_B_3 + 192, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_B11_end = mcycle();
	    }
	}


	// ******************************************************************//
	//                           END THE GEMM                            //
	// ******************************************************************//

	//return to host
	return_to_cva6(SYNC_ALL);

}

