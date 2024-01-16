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

	uint32_t size_setting = gen_size_config((uint8_t)16, (uint8_t)1, (uint8_t)2, (uint8_t)1);

	uint32_t subtraction_setting = gen_subtraction_config(subtraction_a, subtraction_b);

	// Prepare addresses in TCDM for 4 cores


	// TCDM Memory for Cluster0
	int32_t* local_err_0;
	int8_t** local_addr_A;
	int8_t** local_addr_B;
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
		local_addr_A = (int8_t**)(local_err_0 + 1);
		local_addr_B = (int8_t**)(local_addr_A + 4);

		// Space to store matrix
		local_a_0 = (int8_t*)(local_err_0 + 256 * sizeof(int8_t));
		local_b_0 = local_a_0 + 64* sizeof(int8_t);
		local_c_0 = (int32_t*)(local_b_0 + 16384 * sizeof(int8_t));

		C0_ADDR_a = local_a_0;
		C0_ADDR_b = local_b_0;
		C0_ADDR_c = local_c_0;

		*local_err_0 = 0;
		*local_addr_A = A;
		*local_addr_B = B;
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


	// Cluster0 init the GeMM config
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
		if (snrt_is_compute_core()){
		   // Set GEMM configuration CSR 
		   set_batch_gemm(size_setting,
		                  local_a_0, local_b_0,
		                  subtraction_setting,
		                  local_c_0,
		                  256,
		                  256,
		                  256,
		                  512,
		                  512,
		                  512,
		                  1024,
		                  1024,
		                  1024);
		}
	}


	// Cluster1 init the GeMM config
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if (snrt_is_compute_core()){
		   // Set GEMM configuration CSR 
		   set_batch_gemm(size_setting,
		                  local_a_1, local_b_1,
		                  subtraction_setting,
		                  local_c_1,
		                  256,
		                  256,
		                  256,
		                  512,
		                  512,
		                  512,
		                  1024,
		                  1024,
		                  1024);
		}
	}


	// Cluster2 init the GeMM config
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if (snrt_is_compute_core()){
		   // Set GEMM configuration CSR 
		   set_batch_gemm(size_setting,
		                  local_a_2, local_b_2,
		                  subtraction_setting,
		                  local_c_2,
		                  256,
		                  256,
		                  256,
		                  512,
		                  512,
		                  512,
		                  1024,
		                  1024,
		                  1024);
		}
	}


	// Cluster3 init the GeMM config
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if (snrt_is_compute_core()){
		   // Set GEMM configuration CSR 
		   set_batch_gemm(size_setting,
		                  local_a_3, local_b_3,
		                  subtraction_setting,
		                  local_c_3,
		                  256,
		                  256,
		                  256,
		                  512,
		                  512,
		                  512,
		                  1024,
		                  1024,
		                  1024);
		}
	}


	snrt_global_barrier();

    //+----+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    //|    |         | T0  | T1  | T2  | T3  | T4  | T5  | T6  | T7  | T8  | T9  | T10 |
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
    
	// ******************************************************************//
	//                           START THE GEMM                          //
	// ******************************************************************//


	// Cluster 0 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 0, *local_addr_A + 0, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A00_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A01 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A01_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 256, *local_addr_A + 64, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A01_end = mcycle();
	    }
	}



	// C1_load_A00_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_A00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_1, *local_a_C0_1, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_A00_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	    uint32_t dma_C0_load_B00_start = mcycle();
	   snrt_dma_start_2d(local_b_0 + 0, *local_addr_B + 0, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	   snrt_dma_wait_all();
	   uint32_t dma_C0_load_B00_end = mcycle();
	   }
	}



	// C1_load_A01_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_A01_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_1 + 1 * 256, *local_a_C0_1 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_A01_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B10 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	    uint32_t dma_C0_load_B10_start = mcycle();
	   snrt_dma_start_2d(local_b_0 + 256, *local_addr_B + 64, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	   snrt_dma_wait_all();
	   uint32_t dma_C0_load_B10_end = mcycle();
	   }
	}



	// C2_load_B00_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_B00_from_C0_start = mcycle();
			snrt_dma_start_2d(local_b_2, *local_b_C0_2, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_B00_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A10 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A10_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 512, *local_addr_A + 128, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A10_end = mcycle();
	    }
	}



	// C2_load_B10_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_B10_from_C0_start = mcycle();
			snrt_dma_start_2d(local_b_2 + 1 * 256, *local_b_C0_2 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_B10_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A11 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A11_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 768, *local_addr_A + 192, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A11_end = mcycle();
	    }
	}



	// C2_load_A10_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A10_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_2, *local_a_C0_2 + 1 * 512, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A10_from_C0_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B01 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	    uint32_t dma_C0_load_B01_start = mcycle();
	   snrt_dma_start_2d(local_b_0 + 512, *local_addr_B + 128, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	   snrt_dma_wait_all();
	   uint32_t dma_C0_load_B01_end = mcycle();
	   }
	}



	// C2_load_A11_from_C0
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C2_load_A11_from_C0_start = mcycle();
			snrt_dma_start_2d(local_a_2 + 1 * 256, *local_a_C0_2 + 1 * 256 + 1 * 512, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C2_load_A11_from_C0_end = mcycle();
		}
	}


	// C3_load_A10_from_C2
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_A10_from_C2_start = mcycle();
			snrt_dma_start_2d(local_a_3, *local_a_C2_3, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_A10_from_C2_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B11 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	    uint32_t dma_C0_load_B11_start = mcycle();
	   snrt_dma_start_2d(local_b_0 + 768, *local_addr_B + 192, 8 * 8 * sizeof(int8_t), 1024, 256, 16);
	   snrt_dma_wait_all();
	   uint32_t dma_C0_load_B11_end = mcycle();
	   }
	}



	// C1_load_B01_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_B01_from_C0_start = mcycle();
			snrt_dma_start_2d(local_b_1, *local_b_C0_1 + 1 * 512, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_B01_from_C0_end = mcycle();
		}
	}


	// C3_load_A11_from_C2
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_A11_from_C2_start = mcycle();
			snrt_dma_start_2d(local_a_3 + 1 * 256, *local_a_C2_3 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_A11_from_C2_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// C1_load_B11_from_C0
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C1_load_B11_from_C0_start = mcycle();
			snrt_dma_start_2d(local_b_1 + 1 * 256, *local_b_C0_1 + 1 * 256 + 1 * 512, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C1_load_B11_from_C0_end = mcycle();
		}
	}


	// C3_load_B01_from_C1
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_B01_from_C1_start = mcycle();
			snrt_dma_start_2d(local_b_3, *local_b_C1_3, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_B01_from_C1_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//


	// C3_load_B11_from_C1
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C3_load_B11_from_C1_start = mcycle();
			snrt_dma_start_2d(local_b_3 + 1 * 256, *local_b_C1_3 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 16);
			snrt_dma_wait_all();
			uint32_t dma_C3_load_B11_from_C1_end = mcycle();
		}
	}

	// ******************************************************************//
	//                           END THE GEMM                            //
	// ******************************************************************//

	//return to host
	return_to_cva6(SYNC_ALL);

}

