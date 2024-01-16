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

uint8_t*  C4_ADDR_a;
uint8_t*  C4_ADDR_b;
uint32_t* C4_ADDR_c;

uint8_t*  C5_ADDR_a;
uint8_t*  C5_ADDR_b;
uint32_t* C5_ADDR_c;

uint8_t*  C6_ADDR_a;
uint8_t*  C6_ADDR_b;
uint32_t* C6_ADDR_c;

uint8_t*  C7_ADDR_a;
uint8_t*  C7_ADDR_b;
uint32_t* C7_ADDR_c;

uint8_t*  C8_ADDR_a;
uint8_t*  C8_ADDR_b;
uint32_t* C8_ADDR_c;

uint8_t*  C9_ADDR_a;
uint8_t*  C9_ADDR_b;
uint32_t* C9_ADDR_c;

uint8_t*  C10_ADDR_a;
uint8_t*  C10_ADDR_b;
uint32_t* C10_ADDR_c;

uint8_t*  C11_ADDR_a;
uint8_t*  C11_ADDR_b;
uint32_t* C11_ADDR_c;

uint8_t*  C12_ADDR_a;
uint8_t*  C12_ADDR_b;
uint32_t* C12_ADDR_c;

uint8_t*  C13_ADDR_a;
uint8_t*  C13_ADDR_b;
uint32_t* C13_ADDR_c;

uint8_t*  C14_ADDR_a;
uint8_t*  C14_ADDR_b;
uint32_t* C14_ADDR_c;

uint8_t*  C15_ADDR_a;
uint8_t*  C15_ADDR_b;
uint32_t* C15_ADDR_c;

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
	// Group4:SNAX     9         8
	// Group4:SNAX     10         9
	// ---------------------------
	// Group5:SNAX     11         10
	// Group5:SNAX     12         11
	// ---------------------------
	// Group6:SNAX     13         12
	// Group6:SNAX     14         13
	// ---------------------------
	// Group7:SNAX     15         14
	// Group7:SNAX     16         15
	// ---------------------------
	// Group8:SNAX     17         16
	// Group8:SNAX     18         17
	// ---------------------------
	// Group9:SNAX     19         18
	// Group9:SNAX     20         19
	// ---------------------------
	// Group10:SNAX     21         20
	// Group10:SNAX     22         21
	// ---------------------------
	// Group11:SNAX     23         22
	// Group11:SNAX     24         23
	// ---------------------------
	// Group12:SNAX     25         24
	// Group12:SNAX     26         25
	// ---------------------------
	// Group13:SNAX     27         26
	// Group13:SNAX     28         27
	// ---------------------------
	// Group14:SNAX     29         28
	// Group14:SNAX     30         29
	// ---------------------------
	// Group15:SNAX     31         30
	// Group15:SNAX     32         31
	// ---------------------------

	uint32_t group0_bound_lower = 0;
	uint32_t group0_bound_upper = snrt_cluster_core_num();
	uint32_t group1_bound_lower = group0_bound_upper;
	uint32_t group1_bound_upper = 2*snrt_cluster_core_num();
	uint32_t group2_bound_lower = group1_bound_upper;
	uint32_t group2_bound_upper = 3*snrt_cluster_core_num();
	uint32_t group3_bound_lower = group2_bound_upper;
	uint32_t group3_bound_upper = 4*snrt_cluster_core_num();
	uint32_t group4_bound_lower = group3_bound_upper;
	uint32_t group4_bound_upper = 5*snrt_cluster_core_num();
	uint32_t group5_bound_lower = group4_bound_upper;
	uint32_t group5_bound_upper = 6*snrt_cluster_core_num();
	uint32_t group6_bound_lower = group5_bound_upper;
	uint32_t group6_bound_upper = 7*snrt_cluster_core_num();
	uint32_t group7_bound_lower = group6_bound_upper;
	uint32_t group7_bound_upper = 8*snrt_cluster_core_num();
	uint32_t group8_bound_lower = group7_bound_upper;
	uint32_t group8_bound_upper = 9*snrt_cluster_core_num();
	uint32_t group9_bound_lower = group8_bound_upper;
	uint32_t group9_bound_upper = 10*snrt_cluster_core_num();
	uint32_t group10_bound_lower = group9_bound_upper;
	uint32_t group10_bound_upper = 11*snrt_cluster_core_num();
	uint32_t group11_bound_lower = group10_bound_upper;
	uint32_t group11_bound_upper = 12*snrt_cluster_core_num();
	uint32_t group12_bound_lower = group11_bound_upper;
	uint32_t group12_bound_upper = 13*snrt_cluster_core_num();
	uint32_t group13_bound_lower = group12_bound_upper;
	uint32_t group13_bound_upper = 14*snrt_cluster_core_num();
	uint32_t group14_bound_lower = group13_bound_upper;
	uint32_t group14_bound_upper = 15*snrt_cluster_core_num();
	uint32_t group15_bound_lower = group14_bound_upper;
	uint32_t group15_bound_upper = 16*snrt_cluster_core_num();

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


	// TCDM Memory for Cluster4
	int32_t* local_err_4;
	int8_t** local_addr_A_4;
	int8_t** local_addr_B_4;
	int8_t*  local_a_4;
	int8_t*  local_b_4;
	int32_t* local_c_4;


	// TCDM Memory for Cluster5
	int32_t* local_err_5;
	int8_t** local_addr_A_5;
	int8_t** local_addr_B_5;
	int8_t*  local_a_5;
	int8_t*  local_b_5;
	int32_t* local_c_5;


	// TCDM Memory for Cluster6
	int32_t* local_err_6;
	int8_t** local_addr_A_6;
	int8_t** local_addr_B_6;
	int8_t*  local_a_6;
	int8_t*  local_b_6;
	int32_t* local_c_6;


	// TCDM Memory for Cluster7
	int32_t* local_err_7;
	int8_t** local_addr_A_7;
	int8_t** local_addr_B_7;
	int8_t*  local_a_7;
	int8_t*  local_b_7;
	int32_t* local_c_7;


	// TCDM Memory for Cluster8
	int32_t* local_err_8;
	int8_t** local_addr_A_8;
	int8_t** local_addr_B_8;
	int8_t*  local_a_8;
	int8_t*  local_b_8;
	int32_t* local_c_8;


	// TCDM Memory for Cluster9
	int32_t* local_err_9;
	int8_t** local_addr_A_9;
	int8_t** local_addr_B_9;
	int8_t*  local_a_9;
	int8_t*  local_b_9;
	int32_t* local_c_9;


	// TCDM Memory for Cluster10
	int32_t* local_err_10;
	int8_t** local_addr_A_10;
	int8_t** local_addr_B_10;
	int8_t*  local_a_10;
	int8_t*  local_b_10;
	int32_t* local_c_10;


	// TCDM Memory for Cluster11
	int32_t* local_err_11;
	int8_t** local_addr_A_11;
	int8_t** local_addr_B_11;
	int8_t*  local_a_11;
	int8_t*  local_b_11;
	int32_t* local_c_11;


	// TCDM Memory for Cluster12
	int32_t* local_err_12;
	int8_t** local_addr_A_12;
	int8_t** local_addr_B_12;
	int8_t*  local_a_12;
	int8_t*  local_b_12;
	int32_t* local_c_12;


	// TCDM Memory for Cluster13
	int32_t* local_err_13;
	int8_t** local_addr_A_13;
	int8_t** local_addr_B_13;
	int8_t*  local_a_13;
	int8_t*  local_b_13;
	int32_t* local_c_13;


	// TCDM Memory for Cluster14
	int32_t* local_err_14;
	int8_t** local_addr_A_14;
	int8_t** local_addr_B_14;
	int8_t*  local_a_14;
	int8_t*  local_b_14;
	int32_t* local_c_14;


	// TCDM Memory for Cluster15
	int32_t* local_err_15;
	int8_t** local_addr_A_15;
	int8_t** local_addr_B_15;
	int8_t*  local_a_15;
	int8_t*  local_b_15;
	int32_t* local_c_15;


	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
		// Start a new row to store err
		local_err_0 = (int32_t*)snrt_l1_next();
		local_addr_A_0 = (int8_t**)(local_err_0 + 1);
		local_addr_B_0 = (int8_t**)(local_addr_A_0 + 4);

		// Space to store matrix
		local_a_0 = (int8_t*)(local_err_0 + 256 * sizeof(int8_t));
		local_b_0 = local_a_0 + 64* sizeof(int8_t);
		local_c_0 = (int32_t*)(local_b_0 + 32768 * sizeof(int8_t));

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
		local_c_1 = (int32_t*)(local_b_1 + 32768 * sizeof(int8_t));

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
		local_c_2 = (int32_t*)(local_b_2 + 32768 * sizeof(int8_t));

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
		local_c_3 = (int32_t*)(local_b_3 + 32768 * sizeof(int8_t));

		*local_err_3 = 0;
		*local_addr_A_3 = A;
		*local_addr_B_3 = B;
	}


	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		// Start a new row to store err
		local_err_4 = (int32_t*)snrt_l1_next();
		local_addr_A_4 = (int8_t**)(local_err_4 + 1);
		local_addr_B_4 = (int8_t**)(local_addr_A_4 + 4);

		// Space to store matrix
		local_a_4 = (int8_t*)(local_err_4 + 256 * sizeof(int8_t));
		local_b_4 = local_a_4 + 64* sizeof(int8_t);
		local_c_4 = (int32_t*)(local_b_4 + 32768 * sizeof(int8_t));

		*local_err_4 = 0;
		*local_addr_A_4 = A;
		*local_addr_B_4 = B;
	}


	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		// Start a new row to store err
		local_err_5 = (int32_t*)snrt_l1_next();
		local_addr_A_5 = (int8_t**)(local_err_5 + 1);
		local_addr_B_5 = (int8_t**)(local_addr_A_5 + 4);

		// Space to store matrix
		local_a_5 = (int8_t*)(local_err_5 + 256 * sizeof(int8_t));
		local_b_5 = local_a_5 + 64* sizeof(int8_t);
		local_c_5 = (int32_t*)(local_b_5 + 32768 * sizeof(int8_t));

		*local_err_5 = 0;
		*local_addr_A_5 = A;
		*local_addr_B_5 = B;
	}


	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		// Start a new row to store err
		local_err_6 = (int32_t*)snrt_l1_next();
		local_addr_A_6 = (int8_t**)(local_err_6 + 1);
		local_addr_B_6 = (int8_t**)(local_addr_A_6 + 4);

		// Space to store matrix
		local_a_6 = (int8_t*)(local_err_6 + 256 * sizeof(int8_t));
		local_b_6 = local_a_6 + 64* sizeof(int8_t);
		local_c_6 = (int32_t*)(local_b_6 + 32768 * sizeof(int8_t));

		*local_err_6 = 0;
		*local_addr_A_6 = A;
		*local_addr_B_6 = B;
	}


	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		// Start a new row to store err
		local_err_7 = (int32_t*)snrt_l1_next();
		local_addr_A_7 = (int8_t**)(local_err_7 + 1);
		local_addr_B_7 = (int8_t**)(local_addr_A_7 + 4);

		// Space to store matrix
		local_a_7 = (int8_t*)(local_err_7 + 256 * sizeof(int8_t));
		local_b_7 = local_a_7 + 64* sizeof(int8_t);
		local_c_7 = (int32_t*)(local_b_7 + 32768 * sizeof(int8_t));

		*local_err_7 = 0;
		*local_addr_A_7 = A;
		*local_addr_B_7 = B;
	}


	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		// Start a new row to store err
		local_err_8 = (int32_t*)snrt_l1_next();
		local_addr_A_8 = (int8_t**)(local_err_8 + 1);
		local_addr_B_8 = (int8_t**)(local_addr_A_8 + 4);

		// Space to store matrix
		local_a_8 = (int8_t*)(local_err_8 + 256 * sizeof(int8_t));
		local_b_8 = local_a_8 + 64* sizeof(int8_t);
		local_c_8 = (int32_t*)(local_b_8 + 32768 * sizeof(int8_t));

		*local_err_8 = 0;
		*local_addr_A_8 = A;
		*local_addr_B_8 = B;
	}


	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		// Start a new row to store err
		local_err_9 = (int32_t*)snrt_l1_next();
		local_addr_A_9 = (int8_t**)(local_err_9 + 1);
		local_addr_B_9 = (int8_t**)(local_addr_A_9 + 4);

		// Space to store matrix
		local_a_9 = (int8_t*)(local_err_9 + 256 * sizeof(int8_t));
		local_b_9 = local_a_9 + 64* sizeof(int8_t);
		local_c_9 = (int32_t*)(local_b_9 + 32768 * sizeof(int8_t));

		*local_err_9 = 0;
		*local_addr_A_9 = A;
		*local_addr_B_9 = B;
	}


	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		// Start a new row to store err
		local_err_10 = (int32_t*)snrt_l1_next();
		local_addr_A_10 = (int8_t**)(local_err_10 + 1);
		local_addr_B_10 = (int8_t**)(local_addr_A_10 + 4);

		// Space to store matrix
		local_a_10 = (int8_t*)(local_err_10 + 256 * sizeof(int8_t));
		local_b_10 = local_a_10 + 64* sizeof(int8_t);
		local_c_10 = (int32_t*)(local_b_10 + 32768 * sizeof(int8_t));

		*local_err_10 = 0;
		*local_addr_A_10 = A;
		*local_addr_B_10 = B;
	}


	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		// Start a new row to store err
		local_err_11 = (int32_t*)snrt_l1_next();
		local_addr_A_11 = (int8_t**)(local_err_11 + 1);
		local_addr_B_11 = (int8_t**)(local_addr_A_11 + 4);

		// Space to store matrix
		local_a_11 = (int8_t*)(local_err_11 + 256 * sizeof(int8_t));
		local_b_11 = local_a_11 + 64* sizeof(int8_t);
		local_c_11 = (int32_t*)(local_b_11 + 32768 * sizeof(int8_t));

		*local_err_11 = 0;
		*local_addr_A_11 = A;
		*local_addr_B_11 = B;
	}


	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		// Start a new row to store err
		local_err_12 = (int32_t*)snrt_l1_next();
		local_addr_A_12 = (int8_t**)(local_err_12 + 1);
		local_addr_B_12 = (int8_t**)(local_addr_A_12 + 4);

		// Space to store matrix
		local_a_12 = (int8_t*)(local_err_12 + 256 * sizeof(int8_t));
		local_b_12 = local_a_12 + 64* sizeof(int8_t);
		local_c_12 = (int32_t*)(local_b_12 + 32768 * sizeof(int8_t));

		*local_err_12 = 0;
		*local_addr_A_12 = A;
		*local_addr_B_12 = B;
	}


	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		// Start a new row to store err
		local_err_13 = (int32_t*)snrt_l1_next();
		local_addr_A_13 = (int8_t**)(local_err_13 + 1);
		local_addr_B_13 = (int8_t**)(local_addr_A_13 + 4);

		// Space to store matrix
		local_a_13 = (int8_t*)(local_err_13 + 256 * sizeof(int8_t));
		local_b_13 = local_a_13 + 64* sizeof(int8_t);
		local_c_13 = (int32_t*)(local_b_13 + 32768 * sizeof(int8_t));

		*local_err_13 = 0;
		*local_addr_A_13 = A;
		*local_addr_B_13 = B;
	}


	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		// Start a new row to store err
		local_err_14 = (int32_t*)snrt_l1_next();
		local_addr_A_14 = (int8_t**)(local_err_14 + 1);
		local_addr_B_14 = (int8_t**)(local_addr_A_14 + 4);

		// Space to store matrix
		local_a_14 = (int8_t*)(local_err_14 + 256 * sizeof(int8_t));
		local_b_14 = local_a_14 + 64* sizeof(int8_t);
		local_c_14 = (int32_t*)(local_b_14 + 32768 * sizeof(int8_t));

		*local_err_14 = 0;
		*local_addr_A_14 = A;
		*local_addr_B_14 = B;
	}


	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		// Start a new row to store err
		local_err_15 = (int32_t*)snrt_l1_next();
		local_addr_A_15 = (int8_t**)(local_err_15 + 1);
		local_addr_B_15 = (int8_t**)(local_addr_A_15 + 4);

		// Space to store matrix
		local_a_15 = (int8_t*)(local_err_15 + 256 * sizeof(int8_t));
		local_b_15 = local_a_15 + 64* sizeof(int8_t);
		local_c_15 = (int32_t*)(local_b_15 + 32768 * sizeof(int8_t));

		*local_err_15 = 0;
		*local_addr_A_15 = A;
		*local_addr_B_15 = B;
	}


	snrt_global_barrier();

//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//|     | T0  | T1  | T2  | T3  | T4  | T5  | T6  | T7  |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C00 | A00 | A01 | A02 | A03 | B00 | B10 | B20 | B30 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C01 | A00 | A01 | A02 | A03 | B01 | B11 | B21 | B31 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C02 | A00 | A01 | A02 | A03 | B02 | B12 | B22 | B32 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C03 | A00 | A01 | A02 | A03 | B03 | B13 | B23 | B33 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C10 | A10 | A11 | A12 | A13 | B00 | B10 | B20 | B30 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C11 | A10 | A11 | A12 | A13 | B01 | B11 | B21 | B31 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C12 | A10 | A11 | A12 | A13 | B02 | B12 | B22 | B32 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C13 | A10 | A11 | A12 | A13 | B03 | B13 | B23 | B33 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C20 | A20 | A21 | A22 | A23 | B00 | B10 | B20 | B30 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C21 | A20 | A21 | A22 | A23 | B01 | B11 | B21 | B31 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C22 | A20 | A21 | A22 | A23 | B02 | B12 | B22 | B32 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C23 | A20 | A21 | A22 | A23 | B03 | B13 | B23 | B33 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C30 | A30 | A31 | A32 | A33 | B00 | B10 | B20 | B30 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C31 | A30 | A31 | A32 | A33 | B01 | B11 | B21 | B31 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C32 | A30 | A31 | A32 | A33 | B02 | B12 | B22 | B32 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C33 | A30 | A31 | A32 | A33 | B03 | B13 | B23 | B33 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    
	// ******************************************************************//
	//                           START THE GEMM                          //
	// ******************************************************************//


	// Cluster 0 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 0, *local_addr_A_0 + 0, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A00_end = mcycle();
	    }
	}



	// Cluster 1 Load the A00 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_1 + 0, *local_addr_A_1 + 0, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_A00_end = mcycle();
	    }
	}



	// Cluster 2 Load the A00 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_2 + 0, *local_addr_A_2 + 0, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_A00_end = mcycle();
	    }
	}



	// Cluster 3 Load the A00 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_3 + 0, *local_addr_A_3 + 0, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_A00_end = mcycle();
	    }
	}



	// Cluster 4 Load the A10 from L3
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C4_load_A10_start = mcycle();
	    snrt_dma_start_2d(local_a_4 + 1024, *local_addr_A_4 + 256, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C4_load_A10_end = mcycle();
	    }
	}



	// Cluster 5 Load the A10 from L3
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C5_load_A10_start = mcycle();
	    snrt_dma_start_2d(local_a_5 + 1024, *local_addr_A_5 + 256, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C5_load_A10_end = mcycle();
	    }
	}



	// Cluster 6 Load the A10 from L3
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C6_load_A10_start = mcycle();
	    snrt_dma_start_2d(local_a_6 + 1024, *local_addr_A_6 + 256, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C6_load_A10_end = mcycle();
	    }
	}



	// Cluster 7 Load the A10 from L3
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C7_load_A10_start = mcycle();
	    snrt_dma_start_2d(local_a_7 + 1024, *local_addr_A_7 + 256, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C7_load_A10_end = mcycle();
	    }
	}



	// Cluster 8 Load the A20 from L3
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C8_load_A20_start = mcycle();
	    snrt_dma_start_2d(local_a_8 + 2048, *local_addr_A_8 + 512, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C8_load_A20_end = mcycle();
	    }
	}



	// Cluster 9 Load the A20 from L3
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C9_load_A20_start = mcycle();
	    snrt_dma_start_2d(local_a_9 + 2048, *local_addr_A_9 + 512, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C9_load_A20_end = mcycle();
	    }
	}



	// Cluster 10 Load the A20 from L3
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C10_load_A20_start = mcycle();
	    snrt_dma_start_2d(local_a_10 + 2048, *local_addr_A_10 + 512, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C10_load_A20_end = mcycle();
	    }
	}



	// Cluster 11 Load the A20 from L3
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C11_load_A20_start = mcycle();
	    snrt_dma_start_2d(local_a_11 + 2048, *local_addr_A_11 + 512, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C11_load_A20_end = mcycle();
	    }
	}



	// Cluster 12 Load the A30 from L3
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C12_load_A30_start = mcycle();
	    snrt_dma_start_2d(local_a_12 + 3072, *local_addr_A_12 + 768, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C12_load_A30_end = mcycle();
	    }
	}



	// Cluster 13 Load the A30 from L3
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C13_load_A30_start = mcycle();
	    snrt_dma_start_2d(local_a_13 + 3072, *local_addr_A_13 + 768, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C13_load_A30_end = mcycle();
	    }
	}



	// Cluster 14 Load the A30 from L3
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C14_load_A30_start = mcycle();
	    snrt_dma_start_2d(local_a_14 + 3072, *local_addr_A_14 + 768, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C14_load_A30_end = mcycle();
	    }
	}



	// Cluster 15 Load the A30 from L3
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C15_load_A30_start = mcycle();
	    snrt_dma_start_2d(local_a_15 + 3072, *local_addr_A_15 + 768, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C15_load_A30_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A01 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A01_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 256, *local_addr_A_0 + 64, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A01_end = mcycle();
	    }
	}



	// Cluster 1 Load the A01 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_A01_start = mcycle();
	    snrt_dma_start_2d(local_a_1 + 256, *local_addr_A_1 + 64, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_A01_end = mcycle();
	    }
	}



	// Cluster 2 Load the A01 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_A01_start = mcycle();
	    snrt_dma_start_2d(local_a_2 + 256, *local_addr_A_2 + 64, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_A01_end = mcycle();
	    }
	}



	// Cluster 3 Load the A01 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_A01_start = mcycle();
	    snrt_dma_start_2d(local_a_3 + 256, *local_addr_A_3 + 64, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_A01_end = mcycle();
	    }
	}



	// Cluster 4 Load the A11 from L3
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C4_load_A11_start = mcycle();
	    snrt_dma_start_2d(local_a_4 + 1280, *local_addr_A_4 + 320, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C4_load_A11_end = mcycle();
	    }
	}



	// Cluster 5 Load the A11 from L3
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C5_load_A11_start = mcycle();
	    snrt_dma_start_2d(local_a_5 + 1280, *local_addr_A_5 + 320, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C5_load_A11_end = mcycle();
	    }
	}



	// Cluster 6 Load the A11 from L3
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C6_load_A11_start = mcycle();
	    snrt_dma_start_2d(local_a_6 + 1280, *local_addr_A_6 + 320, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C6_load_A11_end = mcycle();
	    }
	}



	// Cluster 7 Load the A11 from L3
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C7_load_A11_start = mcycle();
	    snrt_dma_start_2d(local_a_7 + 1280, *local_addr_A_7 + 320, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C7_load_A11_end = mcycle();
	    }
	}



	// Cluster 8 Load the A21 from L3
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C8_load_A21_start = mcycle();
	    snrt_dma_start_2d(local_a_8 + 2304, *local_addr_A_8 + 576, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C8_load_A21_end = mcycle();
	    }
	}



	// Cluster 9 Load the A21 from L3
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C9_load_A21_start = mcycle();
	    snrt_dma_start_2d(local_a_9 + 2304, *local_addr_A_9 + 576, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C9_load_A21_end = mcycle();
	    }
	}



	// Cluster 10 Load the A21 from L3
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C10_load_A21_start = mcycle();
	    snrt_dma_start_2d(local_a_10 + 2304, *local_addr_A_10 + 576, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C10_load_A21_end = mcycle();
	    }
	}



	// Cluster 11 Load the A21 from L3
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C11_load_A21_start = mcycle();
	    snrt_dma_start_2d(local_a_11 + 2304, *local_addr_A_11 + 576, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C11_load_A21_end = mcycle();
	    }
	}



	// Cluster 12 Load the A31 from L3
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C12_load_A31_start = mcycle();
	    snrt_dma_start_2d(local_a_12 + 3328, *local_addr_A_12 + 832, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C12_load_A31_end = mcycle();
	    }
	}



	// Cluster 13 Load the A31 from L3
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C13_load_A31_start = mcycle();
	    snrt_dma_start_2d(local_a_13 + 3328, *local_addr_A_13 + 832, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C13_load_A31_end = mcycle();
	    }
	}



	// Cluster 14 Load the A31 from L3
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C14_load_A31_start = mcycle();
	    snrt_dma_start_2d(local_a_14 + 3328, *local_addr_A_14 + 832, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C14_load_A31_end = mcycle();
	    }
	}



	// Cluster 15 Load the A31 from L3
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C15_load_A31_start = mcycle();
	    snrt_dma_start_2d(local_a_15 + 3328, *local_addr_A_15 + 832, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C15_load_A31_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A02 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A02_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 512, *local_addr_A_0 + 128, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A02_end = mcycle();
	    }
	}



	// Cluster 1 Load the A02 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_A02_start = mcycle();
	    snrt_dma_start_2d(local_a_1 + 512, *local_addr_A_1 + 128, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_A02_end = mcycle();
	    }
	}



	// Cluster 2 Load the A02 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_A02_start = mcycle();
	    snrt_dma_start_2d(local_a_2 + 512, *local_addr_A_2 + 128, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_A02_end = mcycle();
	    }
	}



	// Cluster 3 Load the A02 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_A02_start = mcycle();
	    snrt_dma_start_2d(local_a_3 + 512, *local_addr_A_3 + 128, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_A02_end = mcycle();
	    }
	}



	// Cluster 4 Load the A12 from L3
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C4_load_A12_start = mcycle();
	    snrt_dma_start_2d(local_a_4 + 1536, *local_addr_A_4 + 384, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C4_load_A12_end = mcycle();
	    }
	}



	// Cluster 5 Load the A12 from L3
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C5_load_A12_start = mcycle();
	    snrt_dma_start_2d(local_a_5 + 1536, *local_addr_A_5 + 384, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C5_load_A12_end = mcycle();
	    }
	}



	// Cluster 6 Load the A12 from L3
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C6_load_A12_start = mcycle();
	    snrt_dma_start_2d(local_a_6 + 1536, *local_addr_A_6 + 384, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C6_load_A12_end = mcycle();
	    }
	}



	// Cluster 7 Load the A12 from L3
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C7_load_A12_start = mcycle();
	    snrt_dma_start_2d(local_a_7 + 1536, *local_addr_A_7 + 384, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C7_load_A12_end = mcycle();
	    }
	}



	// Cluster 8 Load the A22 from L3
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C8_load_A22_start = mcycle();
	    snrt_dma_start_2d(local_a_8 + 2560, *local_addr_A_8 + 640, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C8_load_A22_end = mcycle();
	    }
	}



	// Cluster 9 Load the A22 from L3
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C9_load_A22_start = mcycle();
	    snrt_dma_start_2d(local_a_9 + 2560, *local_addr_A_9 + 640, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C9_load_A22_end = mcycle();
	    }
	}



	// Cluster 10 Load the A22 from L3
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C10_load_A22_start = mcycle();
	    snrt_dma_start_2d(local_a_10 + 2560, *local_addr_A_10 + 640, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C10_load_A22_end = mcycle();
	    }
	}



	// Cluster 11 Load the A22 from L3
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C11_load_A22_start = mcycle();
	    snrt_dma_start_2d(local_a_11 + 2560, *local_addr_A_11 + 640, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C11_load_A22_end = mcycle();
	    }
	}



	// Cluster 12 Load the A32 from L3
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C12_load_A32_start = mcycle();
	    snrt_dma_start_2d(local_a_12 + 3584, *local_addr_A_12 + 896, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C12_load_A32_end = mcycle();
	    }
	}



	// Cluster 13 Load the A32 from L3
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C13_load_A32_start = mcycle();
	    snrt_dma_start_2d(local_a_13 + 3584, *local_addr_A_13 + 896, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C13_load_A32_end = mcycle();
	    }
	}



	// Cluster 14 Load the A32 from L3
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C14_load_A32_start = mcycle();
	    snrt_dma_start_2d(local_a_14 + 3584, *local_addr_A_14 + 896, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C14_load_A32_end = mcycle();
	    }
	}



	// Cluster 15 Load the A32 from L3
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C15_load_A32_start = mcycle();
	    snrt_dma_start_2d(local_a_15 + 3584, *local_addr_A_15 + 896, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C15_load_A32_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the A03 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_A03_start = mcycle();
	    snrt_dma_start_2d(local_a_0 + 768, *local_addr_A_0 + 192, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_A03_end = mcycle();
	    }
	}



	// Cluster 1 Load the A03 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_A03_start = mcycle();
	    snrt_dma_start_2d(local_a_1 + 768, *local_addr_A_1 + 192, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_A03_end = mcycle();
	    }
	}



	// Cluster 2 Load the A03 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_A03_start = mcycle();
	    snrt_dma_start_2d(local_a_2 + 768, *local_addr_A_2 + 192, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_A03_end = mcycle();
	    }
	}



	// Cluster 3 Load the A03 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_A03_start = mcycle();
	    snrt_dma_start_2d(local_a_3 + 768, *local_addr_A_3 + 192, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_A03_end = mcycle();
	    }
	}



	// Cluster 4 Load the A13 from L3
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C4_load_A13_start = mcycle();
	    snrt_dma_start_2d(local_a_4 + 1792, *local_addr_A_4 + 448, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C4_load_A13_end = mcycle();
	    }
	}



	// Cluster 5 Load the A13 from L3
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C5_load_A13_start = mcycle();
	    snrt_dma_start_2d(local_a_5 + 1792, *local_addr_A_5 + 448, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C5_load_A13_end = mcycle();
	    }
	}



	// Cluster 6 Load the A13 from L3
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C6_load_A13_start = mcycle();
	    snrt_dma_start_2d(local_a_6 + 1792, *local_addr_A_6 + 448, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C6_load_A13_end = mcycle();
	    }
	}



	// Cluster 7 Load the A13 from L3
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C7_load_A13_start = mcycle();
	    snrt_dma_start_2d(local_a_7 + 1792, *local_addr_A_7 + 448, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C7_load_A13_end = mcycle();
	    }
	}



	// Cluster 8 Load the A23 from L3
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C8_load_A23_start = mcycle();
	    snrt_dma_start_2d(local_a_8 + 2816, *local_addr_A_8 + 704, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C8_load_A23_end = mcycle();
	    }
	}



	// Cluster 9 Load the A23 from L3
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C9_load_A23_start = mcycle();
	    snrt_dma_start_2d(local_a_9 + 2816, *local_addr_A_9 + 704, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C9_load_A23_end = mcycle();
	    }
	}



	// Cluster 10 Load the A23 from L3
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C10_load_A23_start = mcycle();
	    snrt_dma_start_2d(local_a_10 + 2816, *local_addr_A_10 + 704, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C10_load_A23_end = mcycle();
	    }
	}



	// Cluster 11 Load the A23 from L3
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C11_load_A23_start = mcycle();
	    snrt_dma_start_2d(local_a_11 + 2816, *local_addr_A_11 + 704, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C11_load_A23_end = mcycle();
	    }
	}



	// Cluster 12 Load the A33 from L3
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C12_load_A33_start = mcycle();
	    snrt_dma_start_2d(local_a_12 + 3840, *local_addr_A_12 + 960, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C12_load_A33_end = mcycle();
	    }
	}



	// Cluster 13 Load the A33 from L3
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C13_load_A33_start = mcycle();
	    snrt_dma_start_2d(local_a_13 + 3840, *local_addr_A_13 + 960, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C13_load_A33_end = mcycle();
	    }
	}



	// Cluster 14 Load the A33 from L3
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C14_load_A33_start = mcycle();
	    snrt_dma_start_2d(local_a_14 + 3840, *local_addr_A_14 + 960, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C14_load_A33_end = mcycle();
	    }
	}



	// Cluster 15 Load the A33 from L3
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C15_load_A33_start = mcycle();
	    snrt_dma_start_2d(local_a_15 + 3840, *local_addr_A_15 + 960, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C15_load_A33_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_B00_start = mcycle();
	    snrt_dma_start_2d(local_b_0 + 0, *local_addr_B_0 + 0, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_B00_end = mcycle();
	    }
	}



	// Cluster 1 Load the B01 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_B01_start = mcycle();
	    snrt_dma_start_2d(local_b_1 + 1024, *local_addr_B_1 + 256, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_B01_end = mcycle();
	    }
	}



	// Cluster 2 Load the B02 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_B02_start = mcycle();
	    snrt_dma_start_2d(local_b_2 + 2048, *local_addr_B_2 + 512, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_B02_end = mcycle();
	    }
	}



	// Cluster 3 Load the B03 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_B03_start = mcycle();
	    snrt_dma_start_2d(local_b_3 + 3072, *local_addr_B_3 + 768, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_B03_end = mcycle();
	    }
	}



	// Cluster 4 Load the B00 from L3
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C4_load_B00_start = mcycle();
	    snrt_dma_start_2d(local_b_4 + 0, *local_addr_B_4 + 0, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C4_load_B00_end = mcycle();
	    }
	}



	// Cluster 5 Load the B01 from L3
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C5_load_B01_start = mcycle();
	    snrt_dma_start_2d(local_b_5 + 1024, *local_addr_B_5 + 256, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C5_load_B01_end = mcycle();
	    }
	}



	// Cluster 6 Load the B02 from L3
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C6_load_B02_start = mcycle();
	    snrt_dma_start_2d(local_b_6 + 2048, *local_addr_B_6 + 512, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C6_load_B02_end = mcycle();
	    }
	}



	// Cluster 7 Load the B03 from L3
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C7_load_B03_start = mcycle();
	    snrt_dma_start_2d(local_b_7 + 3072, *local_addr_B_7 + 768, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C7_load_B03_end = mcycle();
	    }
	}



	// Cluster 8 Load the B00 from L3
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C8_load_B00_start = mcycle();
	    snrt_dma_start_2d(local_b_8 + 0, *local_addr_B_8 + 0, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C8_load_B00_end = mcycle();
	    }
	}



	// Cluster 9 Load the B01 from L3
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C9_load_B01_start = mcycle();
	    snrt_dma_start_2d(local_b_9 + 1024, *local_addr_B_9 + 256, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C9_load_B01_end = mcycle();
	    }
	}



	// Cluster 10 Load the B02 from L3
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C10_load_B02_start = mcycle();
	    snrt_dma_start_2d(local_b_10 + 2048, *local_addr_B_10 + 512, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C10_load_B02_end = mcycle();
	    }
	}



	// Cluster 11 Load the B03 from L3
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C11_load_B03_start = mcycle();
	    snrt_dma_start_2d(local_b_11 + 3072, *local_addr_B_11 + 768, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C11_load_B03_end = mcycle();
	    }
	}



	// Cluster 12 Load the B00 from L3
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C12_load_B00_start = mcycle();
	    snrt_dma_start_2d(local_b_12 + 0, *local_addr_B_12 + 0, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C12_load_B00_end = mcycle();
	    }
	}



	// Cluster 13 Load the B01 from L3
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C13_load_B01_start = mcycle();
	    snrt_dma_start_2d(local_b_13 + 1024, *local_addr_B_13 + 256, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C13_load_B01_end = mcycle();
	    }
	}



	// Cluster 14 Load the B02 from L3
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C14_load_B02_start = mcycle();
	    snrt_dma_start_2d(local_b_14 + 2048, *local_addr_B_14 + 512, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C14_load_B02_end = mcycle();
	    }
	}



	// Cluster 15 Load the B03 from L3
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C15_load_B03_start = mcycle();
	    snrt_dma_start_2d(local_b_15 + 3072, *local_addr_B_15 + 768, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C15_load_B03_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B10 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_B10_start = mcycle();
	    snrt_dma_start_2d(local_b_0 + 256, *local_addr_B_0 + 64, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_B10_end = mcycle();
	    }
	}



	// Cluster 1 Load the B11 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_B11_start = mcycle();
	    snrt_dma_start_2d(local_b_1 + 1280, *local_addr_B_1 + 320, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_B11_end = mcycle();
	    }
	}



	// Cluster 2 Load the B12 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_B12_start = mcycle();
	    snrt_dma_start_2d(local_b_2 + 2304, *local_addr_B_2 + 576, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_B12_end = mcycle();
	    }
	}



	// Cluster 3 Load the B13 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_B13_start = mcycle();
	    snrt_dma_start_2d(local_b_3 + 3328, *local_addr_B_3 + 832, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_B13_end = mcycle();
	    }
	}



	// Cluster 4 Load the B10 from L3
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C4_load_B10_start = mcycle();
	    snrt_dma_start_2d(local_b_4 + 256, *local_addr_B_4 + 64, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C4_load_B10_end = mcycle();
	    }
	}



	// Cluster 5 Load the B11 from L3
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C5_load_B11_start = mcycle();
	    snrt_dma_start_2d(local_b_5 + 1280, *local_addr_B_5 + 320, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C5_load_B11_end = mcycle();
	    }
	}



	// Cluster 6 Load the B12 from L3
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C6_load_B12_start = mcycle();
	    snrt_dma_start_2d(local_b_6 + 2304, *local_addr_B_6 + 576, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C6_load_B12_end = mcycle();
	    }
	}



	// Cluster 7 Load the B13 from L3
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C7_load_B13_start = mcycle();
	    snrt_dma_start_2d(local_b_7 + 3328, *local_addr_B_7 + 832, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C7_load_B13_end = mcycle();
	    }
	}



	// Cluster 8 Load the B10 from L3
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C8_load_B10_start = mcycle();
	    snrt_dma_start_2d(local_b_8 + 256, *local_addr_B_8 + 64, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C8_load_B10_end = mcycle();
	    }
	}



	// Cluster 9 Load the B11 from L3
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C9_load_B11_start = mcycle();
	    snrt_dma_start_2d(local_b_9 + 1280, *local_addr_B_9 + 320, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C9_load_B11_end = mcycle();
	    }
	}



	// Cluster 10 Load the B12 from L3
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C10_load_B12_start = mcycle();
	    snrt_dma_start_2d(local_b_10 + 2304, *local_addr_B_10 + 576, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C10_load_B12_end = mcycle();
	    }
	}



	// Cluster 11 Load the B13 from L3
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C11_load_B13_start = mcycle();
	    snrt_dma_start_2d(local_b_11 + 3328, *local_addr_B_11 + 832, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C11_load_B13_end = mcycle();
	    }
	}



	// Cluster 12 Load the B10 from L3
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C12_load_B10_start = mcycle();
	    snrt_dma_start_2d(local_b_12 + 256, *local_addr_B_12 + 64, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C12_load_B10_end = mcycle();
	    }
	}



	// Cluster 13 Load the B11 from L3
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C13_load_B11_start = mcycle();
	    snrt_dma_start_2d(local_b_13 + 1280, *local_addr_B_13 + 320, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C13_load_B11_end = mcycle();
	    }
	}



	// Cluster 14 Load the B12 from L3
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C14_load_B12_start = mcycle();
	    snrt_dma_start_2d(local_b_14 + 2304, *local_addr_B_14 + 576, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C14_load_B12_end = mcycle();
	    }
	}



	// Cluster 15 Load the B13 from L3
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C15_load_B13_start = mcycle();
	    snrt_dma_start_2d(local_b_15 + 3328, *local_addr_B_15 + 832, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C15_load_B13_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B20 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_B20_start = mcycle();
	    snrt_dma_start_2d(local_b_0 + 512, *local_addr_B_0 + 128, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_B20_end = mcycle();
	    }
	}



	// Cluster 1 Load the B21 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_B21_start = mcycle();
	    snrt_dma_start_2d(local_b_1 + 1536, *local_addr_B_1 + 384, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_B21_end = mcycle();
	    }
	}



	// Cluster 2 Load the B22 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_B22_start = mcycle();
	    snrt_dma_start_2d(local_b_2 + 2560, *local_addr_B_2 + 640, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_B22_end = mcycle();
	    }
	}



	// Cluster 3 Load the B23 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_B23_start = mcycle();
	    snrt_dma_start_2d(local_b_3 + 3584, *local_addr_B_3 + 896, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_B23_end = mcycle();
	    }
	}



	// Cluster 4 Load the B20 from L3
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C4_load_B20_start = mcycle();
	    snrt_dma_start_2d(local_b_4 + 512, *local_addr_B_4 + 128, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C4_load_B20_end = mcycle();
	    }
	}



	// Cluster 5 Load the B21 from L3
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C5_load_B21_start = mcycle();
	    snrt_dma_start_2d(local_b_5 + 1536, *local_addr_B_5 + 384, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C5_load_B21_end = mcycle();
	    }
	}



	// Cluster 6 Load the B22 from L3
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C6_load_B22_start = mcycle();
	    snrt_dma_start_2d(local_b_6 + 2560, *local_addr_B_6 + 640, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C6_load_B22_end = mcycle();
	    }
	}



	// Cluster 7 Load the B23 from L3
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C7_load_B23_start = mcycle();
	    snrt_dma_start_2d(local_b_7 + 3584, *local_addr_B_7 + 896, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C7_load_B23_end = mcycle();
	    }
	}



	// Cluster 8 Load the B20 from L3
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C8_load_B20_start = mcycle();
	    snrt_dma_start_2d(local_b_8 + 512, *local_addr_B_8 + 128, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C8_load_B20_end = mcycle();
	    }
	}



	// Cluster 9 Load the B21 from L3
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C9_load_B21_start = mcycle();
	    snrt_dma_start_2d(local_b_9 + 1536, *local_addr_B_9 + 384, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C9_load_B21_end = mcycle();
	    }
	}



	// Cluster 10 Load the B22 from L3
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C10_load_B22_start = mcycle();
	    snrt_dma_start_2d(local_b_10 + 2560, *local_addr_B_10 + 640, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C10_load_B22_end = mcycle();
	    }
	}



	// Cluster 11 Load the B23 from L3
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C11_load_B23_start = mcycle();
	    snrt_dma_start_2d(local_b_11 + 3584, *local_addr_B_11 + 896, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C11_load_B23_end = mcycle();
	    }
	}



	// Cluster 12 Load the B20 from L3
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C12_load_B20_start = mcycle();
	    snrt_dma_start_2d(local_b_12 + 512, *local_addr_B_12 + 128, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C12_load_B20_end = mcycle();
	    }
	}



	// Cluster 13 Load the B21 from L3
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C13_load_B21_start = mcycle();
	    snrt_dma_start_2d(local_b_13 + 1536, *local_addr_B_13 + 384, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C13_load_B21_end = mcycle();
	    }
	}



	// Cluster 14 Load the B22 from L3
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C14_load_B22_start = mcycle();
	    snrt_dma_start_2d(local_b_14 + 2560, *local_addr_B_14 + 640, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C14_load_B22_end = mcycle();
	    }
	}



	// Cluster 15 Load the B23 from L3
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C15_load_B23_start = mcycle();
	    snrt_dma_start_2d(local_b_15 + 3584, *local_addr_B_15 + 896, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C15_load_B23_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//


	// Cluster 0 Load the B30 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C0_load_B30_start = mcycle();
	    snrt_dma_start_2d(local_b_0 + 768, *local_addr_B_0 + 192, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C0_load_B30_end = mcycle();
	    }
	}



	// Cluster 1 Load the B31 from L3
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C1_load_B31_start = mcycle();
	    snrt_dma_start_2d(local_b_1 + 1792, *local_addr_B_1 + 448, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C1_load_B31_end = mcycle();
	    }
	}



	// Cluster 2 Load the B32 from L3
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C2_load_B32_start = mcycle();
	    snrt_dma_start_2d(local_b_2 + 2816, *local_addr_B_2 + 704, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C2_load_B32_end = mcycle();
	    }
	}



	// Cluster 3 Load the B33 from L3
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C3_load_B33_start = mcycle();
	    snrt_dma_start_2d(local_b_3 + 3840, *local_addr_B_3 + 960, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C3_load_B33_end = mcycle();
	    }
	}



	// Cluster 4 Load the B30 from L3
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C4_load_B30_start = mcycle();
	    snrt_dma_start_2d(local_b_4 + 768, *local_addr_B_4 + 192, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C4_load_B30_end = mcycle();
	    }
	}



	// Cluster 5 Load the B31 from L3
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C5_load_B31_start = mcycle();
	    snrt_dma_start_2d(local_b_5 + 1792, *local_addr_B_5 + 448, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C5_load_B31_end = mcycle();
	    }
	}



	// Cluster 6 Load the B32 from L3
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C6_load_B32_start = mcycle();
	    snrt_dma_start_2d(local_b_6 + 2816, *local_addr_B_6 + 704, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C6_load_B32_end = mcycle();
	    }
	}



	// Cluster 7 Load the B33 from L3
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C7_load_B33_start = mcycle();
	    snrt_dma_start_2d(local_b_7 + 3840, *local_addr_B_7 + 960, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C7_load_B33_end = mcycle();
	    }
	}



	// Cluster 8 Load the B30 from L3
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C8_load_B30_start = mcycle();
	    snrt_dma_start_2d(local_b_8 + 768, *local_addr_B_8 + 192, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C8_load_B30_end = mcycle();
	    }
	}



	// Cluster 9 Load the B31 from L3
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C9_load_B31_start = mcycle();
	    snrt_dma_start_2d(local_b_9 + 1792, *local_addr_B_9 + 448, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C9_load_B31_end = mcycle();
	    }
	}



	// Cluster 10 Load the B32 from L3
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C10_load_B32_start = mcycle();
	    snrt_dma_start_2d(local_b_10 + 2816, *local_addr_B_10 + 704, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C10_load_B32_end = mcycle();
	    }
	}



	// Cluster 11 Load the B33 from L3
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C11_load_B33_start = mcycle();
	    snrt_dma_start_2d(local_b_11 + 3840, *local_addr_B_11 + 960, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C11_load_B33_end = mcycle();
	    }
	}



	// Cluster 12 Load the B30 from L3
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C12_load_B30_start = mcycle();
	    snrt_dma_start_2d(local_b_12 + 768, *local_addr_B_12 + 192, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C12_load_B30_end = mcycle();
	    }
	}



	// Cluster 13 Load the B31 from L3
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C13_load_B31_start = mcycle();
	    snrt_dma_start_2d(local_b_13 + 1792, *local_addr_B_13 + 448, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C13_load_B31_end = mcycle();
	    }
	}



	// Cluster 14 Load the B32 from L3
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C14_load_B32_start = mcycle();
	    snrt_dma_start_2d(local_b_14 + 2816, *local_addr_B_14 + 704, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C14_load_B32_end = mcycle();
	    }
	}



	// Cluster 15 Load the B33 from L3
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C15_load_B33_start = mcycle();
	    snrt_dma_start_2d(local_b_15 + 3840, *local_addr_B_15 + 960, 8 * 8 * sizeof(int8_t), 4096, 1024, 4);
	    snrt_dma_wait_all();
	    uint32_t dma_C15_load_B33_end = mcycle();
	    }
	}


	// ******************************************************************//
	//                           END THE GEMM                            //
	// ******************************************************************//

	//return to host
	return_to_cva6(SYNC_ALL);

}

