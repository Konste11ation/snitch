#include "data.h"
#include "snax-gemm-lib.h"
#include "snax-gemm-params.h"

uint8_t*  C00_ADDR_a = 999;
uint8_t*  C00_ADDR_b = 999;
uint8_t*  C01_ADDR_a = 999;
uint8_t*  C01_ADDR_b = 999;
uint8_t*  C02_ADDR_a = 999;
uint8_t*  C02_ADDR_b = 999;
uint8_t*  C03_ADDR_a = 999;
uint8_t*  C03_ADDR_b = 999;
uint8_t*  C10_ADDR_a = 999;
uint8_t*  C10_ADDR_b = 999;
uint8_t*  C11_ADDR_a = 999;
uint8_t*  C11_ADDR_b = 999;
uint8_t*  C12_ADDR_a = 999;
uint8_t*  C12_ADDR_b = 999;
uint8_t*  C13_ADDR_a = 999;
uint8_t*  C13_ADDR_b = 999;
uint8_t*  C20_ADDR_a = 999;
uint8_t*  C20_ADDR_b = 999;
uint8_t*  C21_ADDR_a = 999;
uint8_t*  C21_ADDR_b = 999;
uint8_t*  C22_ADDR_a = 999;
uint8_t*  C22_ADDR_b = 999;
uint8_t*  C23_ADDR_a = 999;
uint8_t*  C23_ADDR_b = 999;
uint8_t*  C30_ADDR_a = 999;
uint8_t*  C30_ADDR_b = 999;
uint8_t*  C31_ADDR_a = 999;
uint8_t*  C31_ADDR_b = 999;
uint8_t*  C32_ADDR_a = 999;
uint8_t*  C32_ADDR_b = 999;
uint8_t*  C33_ADDR_a = 999;
uint8_t*  C33_ADDR_b = 999;

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

	// Prepare addresses in TCDM


	// TCDM Memory for Cluster00
	int32_t* local_err_C00;
	int8_t*  local_a_C00;
	int8_t*  local_b_C00;
	int32_t* local_c_C00;
	int8_t** local_addr_A_C00;
	int8_t** local_addr_B_C00;

	// TCDM Memory for Cluster01
	int32_t* local_err_C01;
	int8_t*  local_a_C01;
	int8_t*  local_b_C01;
	int32_t* local_c_C01;
	int8_t** local_a_C00_C01;
	int8_t** local_b_C00_C01;
	int32_t** local_c_C00_C01;

	// TCDM Memory for Cluster02
	int32_t* local_err_C02;
	int8_t*  local_a_C02;
	int8_t*  local_b_C02;
	int32_t* local_c_C02;
	int8_t** local_a_C01_C02;
	int8_t** local_b_C01_C02;
	int32_t** local_c_C01_C02;

	// TCDM Memory for Cluster03
	int32_t* local_err_C03;
	int8_t*  local_a_C03;
	int8_t*  local_b_C03;
	int32_t* local_c_C03;
	int8_t** local_a_C02_C03;
	int8_t** local_b_C02_C03;
	int32_t** local_c_C02_C03;

	// TCDM Memory for Cluster10
	int32_t* local_err_C10;
	int8_t*  local_a_C10;
	int8_t*  local_b_C10;
	int32_t* local_c_C10;
	int8_t** local_a_C00_C10;
	int8_t** local_b_C00_C10;
	int32_t** local_c_C00_C10;

	// TCDM Memory for Cluster11
	int32_t* local_err_C11;
	int8_t*  local_a_C11;
	int8_t*  local_b_C11;
	int32_t* local_c_C11;
	int8_t** local_a_C10_C11;
	int8_t** local_b_C10_C11;
	int32_t** local_c_C10_C11;
	int8_t** local_a_C01_C11;
	int8_t** local_b_C01_C11;
	int32_t** local_c_C01_C11;

	// TCDM Memory for Cluster12
	int32_t* local_err_C12;
	int8_t*  local_a_C12;
	int8_t*  local_b_C12;
	int32_t* local_c_C12;
	int8_t** local_a_C11_C12;
	int8_t** local_b_C11_C12;
	int32_t** local_c_C11_C12;
	int8_t** local_a_C02_C12;
	int8_t** local_b_C02_C12;
	int32_t** local_c_C02_C12;

	// TCDM Memory for Cluster13
	int32_t* local_err_C13;
	int8_t*  local_a_C13;
	int8_t*  local_b_C13;
	int32_t* local_c_C13;
	int8_t** local_a_C12_C13;
	int8_t** local_b_C12_C13;
	int32_t** local_c_C12_C13;
	int8_t** local_a_C03_C13;
	int8_t** local_b_C03_C13;
	int32_t** local_c_C03_C13;

	// TCDM Memory for Cluster20
	int32_t* local_err_C20;
	int8_t*  local_a_C20;
	int8_t*  local_b_C20;
	int32_t* local_c_C20;
	int8_t** local_a_C10_C20;
	int8_t** local_b_C10_C20;
	int32_t** local_c_C10_C20;

	// TCDM Memory for Cluster21
	int32_t* local_err_C21;
	int8_t*  local_a_C21;
	int8_t*  local_b_C21;
	int32_t* local_c_C21;
	int8_t** local_a_C20_C21;
	int8_t** local_b_C20_C21;
	int32_t** local_c_C20_C21;
	int8_t** local_a_C11_C21;
	int8_t** local_b_C11_C21;
	int32_t** local_c_C11_C21;

	// TCDM Memory for Cluster22
	int32_t* local_err_C22;
	int8_t*  local_a_C22;
	int8_t*  local_b_C22;
	int32_t* local_c_C22;
	int8_t** local_a_C21_C22;
	int8_t** local_b_C21_C22;
	int32_t** local_c_C21_C22;
	int8_t** local_a_C12_C22;
	int8_t** local_b_C12_C22;
	int32_t** local_c_C12_C22;

	// TCDM Memory for Cluster23
	int32_t* local_err_C23;
	int8_t*  local_a_C23;
	int8_t*  local_b_C23;
	int32_t* local_c_C23;
	int8_t** local_a_C22_C23;
	int8_t** local_b_C22_C23;
	int32_t** local_c_C22_C23;
	int8_t** local_a_C13_C23;
	int8_t** local_b_C13_C23;
	int32_t** local_c_C13_C23;

	// TCDM Memory for Cluster30
	int32_t* local_err_C30;
	int8_t*  local_a_C30;
	int8_t*  local_b_C30;
	int32_t* local_c_C30;
	int8_t** local_a_C20_C30;
	int8_t** local_b_C20_C30;
	int32_t** local_c_C20_C30;

	// TCDM Memory for Cluster31
	int32_t* local_err_C31;
	int8_t*  local_a_C31;
	int8_t*  local_b_C31;
	int32_t* local_c_C31;
	int8_t** local_a_C30_C31;
	int8_t** local_b_C30_C31;
	int32_t** local_c_C30_C31;
	int8_t** local_a_C21_C31;
	int8_t** local_b_C21_C31;
	int32_t** local_c_C21_C31;

	// TCDM Memory for Cluster32
	int32_t* local_err_C32;
	int8_t*  local_a_C32;
	int8_t*  local_b_C32;
	int32_t* local_c_C32;
	int8_t** local_a_C31_C32;
	int8_t** local_b_C31_C32;
	int32_t** local_c_C31_C32;
	int8_t** local_a_C22_C32;
	int8_t** local_b_C22_C32;
	int32_t** local_c_C22_C32;

	// TCDM Memory for Cluster33
	int32_t* local_err_C33;
	int8_t*  local_a_C33;
	int8_t*  local_b_C33;
	int32_t* local_c_C33;
	int8_t** local_a_C32_C33;
	int8_t** local_b_C32_C33;
	int32_t** local_c_C32_C33;
	int8_t** local_a_C23_C33;
	int8_t** local_b_C23_C33;
	int32_t** local_c_C23_C33;
	// Init Addresses in TCDM


	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
		// Start a new row to store err
		local_err_C00 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C00 = (int8_t*)(local_err_C00 + 256 * sizeof(int8_t));
		local_b_C00 = local_a_C00 + 64 * sizeof(int8_t);
		local_c_C00 = (int32_t*)(local_b_C00 + 32768 * sizeof(int8_t));

		local_addr_A_C00 = (int8_t**)(local_err_C00 + 1);
		local_addr_B_C00 = (int8_t**)(local_addr_A_C00 + 1);

		*local_err_C00 = 0;
		*local_addr_A_C00 = A;
		*local_addr_B_C00 = B;
		C00_ADDR_a = local_a_C00;
		C00_ADDR_b = local_b_C00;
	}

	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		// Start a new row to store err
		local_err_C01 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C01 = (int8_t*)(local_err_C01 + 256 * sizeof(int8_t));
		local_b_C01 = local_a_C01 + 64 * sizeof(int8_t);
		local_c_C01 = (int32_t*)(local_b_C01 + 32768 * sizeof(int8_t));

		local_a_C00_C01 = (int8_t**)(local_err_C01 + 1);
		local_b_C00_C01 = (int8_t**)(local_a_C00_C01 + 1);

		*local_err_C01 = 0;
		C01_ADDR_a = local_a_C01;
		C01_ADDR_b = local_b_C01;
	}

	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		// Start a new row to store err
		local_err_C02 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C02 = (int8_t*)(local_err_C02 + 256 * sizeof(int8_t));
		local_b_C02 = local_a_C02 + 64 * sizeof(int8_t);
		local_c_C02 = (int32_t*)(local_b_C02 + 32768 * sizeof(int8_t));

		local_a_C01_C02 = (int8_t**)(local_err_C02 + 1);
		local_b_C01_C02 = (int8_t**)(local_a_C01_C02 + 1);

		*local_err_C02 = 0;
		C02_ADDR_a = local_a_C02;
		C02_ADDR_b = local_b_C02;
	}

	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		// Start a new row to store err
		local_err_C03 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C03 = (int8_t*)(local_err_C03 + 256 * sizeof(int8_t));
		local_b_C03 = local_a_C03 + 64 * sizeof(int8_t);
		local_c_C03 = (int32_t*)(local_b_C03 + 32768 * sizeof(int8_t));

		local_a_C02_C03 = (int8_t**)(local_err_C03 + 1);
		local_b_C02_C03 = (int8_t**)(local_a_C02_C03 + 1);

		*local_err_C03 = 0;
		C03_ADDR_a = local_a_C03;
		C03_ADDR_b = local_b_C03;
	}

	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		// Start a new row to store err
		local_err_C10 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C10 = (int8_t*)(local_err_C10 + 256 * sizeof(int8_t));
		local_b_C10 = local_a_C10 + 64 * sizeof(int8_t);
		local_c_C10 = (int32_t*)(local_b_C10 + 32768 * sizeof(int8_t));

		local_a_C00_C10 = (int8_t**)(local_err_C10 + 1);
		local_b_C00_C10 = (int8_t**)(local_a_C00_C10 + 1);

		*local_err_C10 = 0;
		C10_ADDR_a = local_a_C10;
		C10_ADDR_b = local_b_C10;
	}

	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		// Start a new row to store err
		local_err_C11 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C11 = (int8_t*)(local_err_C11 + 256 * sizeof(int8_t));
		local_b_C11 = local_a_C11 + 64 * sizeof(int8_t);
		local_c_C11 = (int32_t*)(local_b_C11 + 32768 * sizeof(int8_t));

		local_a_C10_C11 = (int8_t**)(local_err_C11 + 1);
		local_b_C10_C11 = (int8_t**)(local_a_C10_C11 + 1);
		local_a_C01_C11 = (int8_t**)(local_b_C10_C11 + 1);
		local_b_C01_C11 = (int8_t**)(local_a_C01_C11 + 1);

		*local_err_C11 = 0;
		C11_ADDR_a = local_a_C11;
		C11_ADDR_b = local_b_C11;
	}

	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		// Start a new row to store err
		local_err_C12 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C12 = (int8_t*)(local_err_C12 + 256 * sizeof(int8_t));
		local_b_C12 = local_a_C12 + 64 * sizeof(int8_t);
		local_c_C12 = (int32_t*)(local_b_C12 + 32768 * sizeof(int8_t));

		local_a_C11_C12 = (int8_t**)(local_err_C12 + 1);
		local_b_C11_C12 = (int8_t**)(local_a_C11_C12 + 1);
		local_a_C02_C12 = (int8_t**)(local_b_C11_C12 + 1);
		local_b_C02_C12 = (int8_t**)(local_a_C02_C12 + 1);

		*local_err_C12 = 0;
		C12_ADDR_a = local_a_C12;
		C12_ADDR_b = local_b_C12;
	}

	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		// Start a new row to store err
		local_err_C13 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C13 = (int8_t*)(local_err_C13 + 256 * sizeof(int8_t));
		local_b_C13 = local_a_C13 + 64 * sizeof(int8_t);
		local_c_C13 = (int32_t*)(local_b_C13 + 32768 * sizeof(int8_t));

		local_a_C12_C13 = (int8_t**)(local_err_C13 + 1);
		local_b_C12_C13 = (int8_t**)(local_a_C12_C13 + 1);
		local_a_C03_C13 = (int8_t**)(local_b_C12_C13 + 1);
		local_b_C03_C13 = (int8_t**)(local_a_C03_C13 + 1);

		*local_err_C13 = 0;
		C13_ADDR_a = local_a_C13;
		C13_ADDR_b = local_b_C13;
	}

	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		// Start a new row to store err
		local_err_C20 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C20 = (int8_t*)(local_err_C20 + 256 * sizeof(int8_t));
		local_b_C20 = local_a_C20 + 64 * sizeof(int8_t);
		local_c_C20 = (int32_t*)(local_b_C20 + 32768 * sizeof(int8_t));

		local_a_C10_C20 = (int8_t**)(local_err_C20 + 1);
		local_b_C10_C20 = (int8_t**)(local_a_C10_C20 + 1);

		*local_err_C20 = 0;
		C20_ADDR_a = local_a_C20;
		C20_ADDR_b = local_b_C20;
	}

	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		// Start a new row to store err
		local_err_C21 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C21 = (int8_t*)(local_err_C21 + 256 * sizeof(int8_t));
		local_b_C21 = local_a_C21 + 64 * sizeof(int8_t);
		local_c_C21 = (int32_t*)(local_b_C21 + 32768 * sizeof(int8_t));

		local_a_C20_C21 = (int8_t**)(local_err_C21 + 1);
		local_b_C20_C21 = (int8_t**)(local_a_C20_C21 + 1);
		local_a_C11_C21 = (int8_t**)(local_b_C20_C21 + 1);
		local_b_C11_C21 = (int8_t**)(local_a_C11_C21 + 1);

		*local_err_C21 = 0;
		C21_ADDR_a = local_a_C21;
		C21_ADDR_b = local_b_C21;
	}

	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		// Start a new row to store err
		local_err_C22 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C22 = (int8_t*)(local_err_C22 + 256 * sizeof(int8_t));
		local_b_C22 = local_a_C22 + 64 * sizeof(int8_t);
		local_c_C22 = (int32_t*)(local_b_C22 + 32768 * sizeof(int8_t));

		local_a_C21_C22 = (int8_t**)(local_err_C22 + 1);
		local_b_C21_C22 = (int8_t**)(local_a_C21_C22 + 1);
		local_a_C12_C22 = (int8_t**)(local_b_C21_C22 + 1);
		local_b_C12_C22 = (int8_t**)(local_a_C12_C22 + 1);

		*local_err_C22 = 0;
		C22_ADDR_a = local_a_C22;
		C22_ADDR_b = local_b_C22;
	}

	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		// Start a new row to store err
		local_err_C23 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C23 = (int8_t*)(local_err_C23 + 256 * sizeof(int8_t));
		local_b_C23 = local_a_C23 + 64 * sizeof(int8_t);
		local_c_C23 = (int32_t*)(local_b_C23 + 32768 * sizeof(int8_t));

		local_a_C22_C23 = (int8_t**)(local_err_C23 + 1);
		local_b_C22_C23 = (int8_t**)(local_a_C22_C23 + 1);
		local_a_C13_C23 = (int8_t**)(local_b_C22_C23 + 1);
		local_b_C13_C23 = (int8_t**)(local_a_C13_C23 + 1);

		*local_err_C23 = 0;
		C23_ADDR_a = local_a_C23;
		C23_ADDR_b = local_b_C23;
	}

	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		// Start a new row to store err
		local_err_C30 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C30 = (int8_t*)(local_err_C30 + 256 * sizeof(int8_t));
		local_b_C30 = local_a_C30 + 64 * sizeof(int8_t);
		local_c_C30 = (int32_t*)(local_b_C30 + 32768 * sizeof(int8_t));

		local_a_C20_C30 = (int8_t**)(local_err_C30 + 1);
		local_b_C20_C30 = (int8_t**)(local_a_C20_C30 + 1);

		*local_err_C30 = 0;
		C30_ADDR_a = local_a_C30;
		C30_ADDR_b = local_b_C30;
	}

	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		// Start a new row to store err
		local_err_C31 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C31 = (int8_t*)(local_err_C31 + 256 * sizeof(int8_t));
		local_b_C31 = local_a_C31 + 64 * sizeof(int8_t);
		local_c_C31 = (int32_t*)(local_b_C31 + 32768 * sizeof(int8_t));

		local_a_C30_C31 = (int8_t**)(local_err_C31 + 1);
		local_b_C30_C31 = (int8_t**)(local_a_C30_C31 + 1);
		local_a_C21_C31 = (int8_t**)(local_b_C30_C31 + 1);
		local_b_C21_C31 = (int8_t**)(local_a_C21_C31 + 1);

		*local_err_C31 = 0;
		C31_ADDR_a = local_a_C31;
		C31_ADDR_b = local_b_C31;
	}

	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		// Start a new row to store err
		local_err_C32 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C32 = (int8_t*)(local_err_C32 + 256 * sizeof(int8_t));
		local_b_C32 = local_a_C32 + 64 * sizeof(int8_t);
		local_c_C32 = (int32_t*)(local_b_C32 + 32768 * sizeof(int8_t));

		local_a_C31_C32 = (int8_t**)(local_err_C32 + 1);
		local_b_C31_C32 = (int8_t**)(local_a_C31_C32 + 1);
		local_a_C22_C32 = (int8_t**)(local_b_C31_C32 + 1);
		local_b_C22_C32 = (int8_t**)(local_a_C22_C32 + 1);

		*local_err_C32 = 0;
		C32_ADDR_a = local_a_C32;
		C32_ADDR_b = local_b_C32;
	}

	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		// Start a new row to store err
		local_err_C33 = (int32_t*)snrt_l1_next();

		// Space to store matrix
		local_a_C33 = (int8_t*)(local_err_C33 + 256 * sizeof(int8_t));
		local_b_C33 = local_a_C33 + 64 * sizeof(int8_t);
		local_c_C33 = (int32_t*)(local_b_C33 + 32768 * sizeof(int8_t));

		local_a_C32_C33 = (int8_t**)(local_err_C33 + 1);
		local_b_C32_C33 = (int8_t**)(local_a_C32_C33 + 1);
		local_a_C23_C33 = (int8_t**)(local_b_C32_C33 + 1);
		local_b_C23_C33 = (int8_t**)(local_a_C23_C33 + 1);

		*local_err_C33 = 0;
		C33_ADDR_a = local_a_C33;
		C33_ADDR_b = local_b_C33;
	}

	snrt_global_barrier();

	//Start the broadcasting of the tcdm address
	//Cluster00 already got the matrix A and B addr

	//Cluster01 gets the TCDM address of the Cluster00

	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
	   if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C00_C01,&C00_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C00_C01,&C00_ADDR_b,4);
	     snrt_dma_wait_all();
	     }
	     snrt_cluster_hw_barrier();
	}
	//Cluster02 gets the TCDM address of the Cluster01

	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
	   if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C01_C02,&C01_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C01_C02,&C01_ADDR_b,4);
	     snrt_dma_wait_all();
	     }
	     snrt_cluster_hw_barrier();
	}
	//Cluster03 gets the TCDM address of the Cluster02

	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
	   if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C02_C03,&C02_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C02_C03,&C02_ADDR_b,4);
	     snrt_dma_wait_all();
	     }
	     snrt_cluster_hw_barrier();
	}
	//Cluster10 gets the TCDM address of the Cluster00

	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
	   if(snrt_is_dm_core()){
	   snrt_dma_start_1d(local_a_C00_C10,&C00_ADDR_a,4);
	   snrt_dma_start_1d(local_b_C00_C10,&C00_ADDR_b,4);
	   snrt_dma_wait_all();
	    }
	    snrt_cluster_hw_barrier();
	}
	//Cluster11 gets the TCDM address of the Cluster01 and Cluster10

	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
	  if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C01_C11,&C01_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C01_C11,&C01_ADDR_b,4);
	     snrt_dma_start_1d(local_a_C10_C11,&C10_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C10_C11,&C10_ADDR_b,4);
	     snrt_dma_wait_all();
	   }
	   snrt_cluster_hw_barrier();
	}
	//Cluster12 gets the TCDM address of the Cluster02 and Cluster11

	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
	  if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C02_C12,&C02_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C02_C12,&C02_ADDR_b,4);
	     snrt_dma_start_1d(local_a_C11_C12,&C11_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C11_C12,&C11_ADDR_b,4);
	     snrt_dma_wait_all();
	   }
	   snrt_cluster_hw_barrier();
	}
	//Cluster13 gets the TCDM address of the Cluster03 and Cluster12

	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
	  if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C03_C13,&C03_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C03_C13,&C03_ADDR_b,4);
	     snrt_dma_start_1d(local_a_C12_C13,&C12_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C12_C13,&C12_ADDR_b,4);
	     snrt_dma_wait_all();
	   }
	   snrt_cluster_hw_barrier();
	}
	//Cluster20 gets the TCDM address of the Cluster10

	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
	   if(snrt_is_dm_core()){
	   snrt_dma_start_1d(local_a_C10_C20,&C10_ADDR_a,4);
	   snrt_dma_start_1d(local_b_C10_C20,&C10_ADDR_b,4);
	   snrt_dma_wait_all();
	    }
	    snrt_cluster_hw_barrier();
	}
	//Cluster21 gets the TCDM address of the Cluster11 and Cluster20

	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
	  if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C11_C21,&C11_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C11_C21,&C11_ADDR_b,4);
	     snrt_dma_start_1d(local_a_C20_C21,&C20_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C20_C21,&C20_ADDR_b,4);
	     snrt_dma_wait_all();
	   }
	   snrt_cluster_hw_barrier();
	}
	//Cluster22 gets the TCDM address of the Cluster12 and Cluster21

	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
	  if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C12_C22,&C12_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C12_C22,&C12_ADDR_b,4);
	     snrt_dma_start_1d(local_a_C21_C22,&C21_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C21_C22,&C21_ADDR_b,4);
	     snrt_dma_wait_all();
	   }
	   snrt_cluster_hw_barrier();
	}
	//Cluster23 gets the TCDM address of the Cluster13 and Cluster22

	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
	  if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C13_C23,&C13_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C13_C23,&C13_ADDR_b,4);
	     snrt_dma_start_1d(local_a_C22_C23,&C22_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C22_C23,&C22_ADDR_b,4);
	     snrt_dma_wait_all();
	   }
	   snrt_cluster_hw_barrier();
	}
	//Cluster30 gets the TCDM address of the Cluster20

	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
	   if(snrt_is_dm_core()){
	   snrt_dma_start_1d(local_a_C20_C30,&C20_ADDR_a,4);
	   snrt_dma_start_1d(local_b_C20_C30,&C20_ADDR_b,4);
	   snrt_dma_wait_all();
	    }
	    snrt_cluster_hw_barrier();
	}
	//Cluster31 gets the TCDM address of the Cluster21 and Cluster30

	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
	  if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C21_C31,&C21_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C21_C31,&C21_ADDR_b,4);
	     snrt_dma_start_1d(local_a_C30_C31,&C30_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C30_C31,&C30_ADDR_b,4);
	     snrt_dma_wait_all();
	   }
	   snrt_cluster_hw_barrier();
	}
	//Cluster32 gets the TCDM address of the Cluster22 and Cluster31

	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
	  if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C22_C32,&C22_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C22_C32,&C22_ADDR_b,4);
	     snrt_dma_start_1d(local_a_C31_C32,&C31_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C31_C32,&C31_ADDR_b,4);
	     snrt_dma_wait_all();
	   }
	   snrt_cluster_hw_barrier();
	}
	//Cluster33 gets the TCDM address of the Cluster23 and Cluster32

	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
	  if(snrt_is_dm_core()){
	     snrt_dma_start_1d(local_a_C23_C33,&C23_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C23_C33,&C23_ADDR_b,4);
	     snrt_dma_start_1d(local_a_C32_C33,&C32_ADDR_a,4);
	     snrt_dma_start_1d(local_b_C32_C33,&C32_ADDR_b,4);
	     snrt_dma_wait_all();
	   }
	   snrt_cluster_hw_barrier();
	}

	snrt_global_barrier();

//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//|     | T0  | T1  | T2  | T3  | T4  | T5  | T6  | T7  | T8  | T9  | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 | T20 | T21 | T22 | T23 | T24 | T25 | T26 | T27 | T28 | T29 | T30 | T31 | T32 | T33 | T34 | T35 | T36 | T37 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C00 | A00 | A01 | A02 | A03 | B00 | B10 | B20 | B30 | A10 | A11 | A12 | A13 | B01 | B11 | B21 | B31 | A20 | A21 | A22 | A23 | B02 | B12 | B22 | B32 | A30 | A31 | A32 | A33 | B03 | B13 | B23 | B33 |     |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C01 |     | A00 | A01 | A02 | A03 |     |     |     |     |     |     |     |     | B01 | B11 | B21 | B31 |     |     |     |     | B02 | B12 | B22 | B32 |     |     |     |     | B03 | B13 | B23 | B33 |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C02 |     |     | A00 | A01 | A02 | A03 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | B02 | B12 | B22 | B32 |     |     |     |     | B03 | B13 | B23 | B33 |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C03 |     |     |     | A00 | A01 | A02 | A03 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | B03 | B13 | B23 | B33 |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C10 |     |     |     |     |     | B00 | B10 | B20 | B30 | A10 | A11 | A12 | A13 |     |     |     |     | A20 | A21 | A22 | A23 |     |     |     |     | A30 | A31 | A32 | A33 |     |     |     |     |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C11 |     |     |     |     |     |     |     |     |     |     | A10 | A11 | A12 | A13 | B01 | B11 | B21 | B31 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C12 |     |     |     |     |     |     |     |     |     |     |     | A10 | A11 | A12 | A13 |     |     |     |     |     |     |     |     | B02 | B12 | B22 | B32 |     |     |     |     |     |     |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C13 |     |     |     |     |     |     |     |     |     |     |     |     | A10 | A11 | A12 | A13 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | B03 | B13 | B23 | B33 |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C20 |     |     |     |     |     |     | B00 | B10 | B20 | B30 |     |     |     |     |     |     |     |     | A20 | A21 | A22 | A23 |     |     |     |     | A30 | A31 | A32 | A33 |     |     |     |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C21 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | B01 | B11 | B21 | B31 | A20 | A21 | A22 | A23 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C22 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | A20 | A21 | A22 | A23 | B02 | B12 | B22 | B32 |     |     |     |     |     |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C23 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | A20 | A21 | A22 | A23 |     |     |     |     |     |     |     |     | B03 | B13 | B23 | B33 |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C30 |     |     |     |     |     |     |     | B00 | B10 | B20 | B30 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | A30 | A31 | A32 | A33 |     |     |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C31 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | B01 | B11 | B21 | B31 |     |     |     |     |     |     |     |     | A30 | A31 | A32 | A33 |     |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C32 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | B02 | B12 | B22 | B32 | A30 | A31 | A32 | A33 |     |     |     |     |     |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//| C33 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     | A30 | A31 | A32 | A33 | B03 | B13 | B23 | B33 |
//+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    
	// ******************************************************************//
	//                           START THE GEMM                          //
	// ******************************************************************//
	//T0:


	// Cluster 00 Load the A00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A00_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 0 * 256, *local_addr_A_C00 + 0, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A00_end = mcycle();
	    }
	}


	snrt_global_barrier();
	// ******************************************************************//
	//T1:


	// Cluster 00 Load the A01 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A01_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 1 * 256, *local_addr_A_C00 + 64, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A01_end = mcycle();
	    }
	}



	// C01_load_A00_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_A00_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C01 + 0 * 256, *local_a_C00_C01 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_A00_from_C00_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T2:


	// Cluster 00 Load the A02 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A02_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 2 * 256, *local_addr_A_C00 + 128, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A02_end = mcycle();
	    }
	}



	// C01_load_A01_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_A01_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C01 + 1 * 256, *local_a_C00_C01 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_A01_from_C00_end = mcycle();
		}
	}


	// C02_load_A00_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_A00_from_C01_start = mcycle();
			snrt_dma_start_2d(local_a_C02 + 0 * 256, *local_a_C01_C02 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_A00_from_C01_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T3:


	// Cluster 00 Load the A03 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A03_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 3 * 256, *local_addr_A_C00 + 192, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A03_end = mcycle();
	    }
	}



	// C01_load_A02_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_A02_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C01 + 2 * 256, *local_a_C00_C01 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_A02_from_C00_end = mcycle();
		}
	}


	// C02_load_A01_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_A01_from_C01_start = mcycle();
			snrt_dma_start_2d(local_a_C02 + 1 * 256, *local_a_C01_C02 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_A01_from_C01_end = mcycle();
		}
	}


	// C03_load_A00_from_C02
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C03_load_A00_from_C02_start = mcycle();
			snrt_dma_start_2d(local_a_C03 + 0 * 256, *local_a_C02_C03 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C03_load_A00_from_C02_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T4:


	// Cluster 00 Load the B00 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B00_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 0 * 256, *local_addr_B_C00 + 0, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B00_end = mcycle();
	    }
	}



	// C01_load_A03_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_A03_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C01 + 3 * 256, *local_a_C00_C01 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_A03_from_C00_end = mcycle();
		}
	}


	// C02_load_A02_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_A02_from_C01_start = mcycle();
			snrt_dma_start_2d(local_a_C02 + 2 * 256, *local_a_C01_C02 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_A02_from_C01_end = mcycle();
		}
	}


	// C03_load_A01_from_C02
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C03_load_A01_from_C02_start = mcycle();
			snrt_dma_start_2d(local_a_C03 + 1 * 256, *local_a_C02_C03 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C03_load_A01_from_C02_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T5:


	// Cluster 00 Load the B10 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B10_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 1 * 256, *local_addr_B_C00 + 64, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B10_end = mcycle();
	    }
	}



	// C02_load_A03_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_A03_from_C01_start = mcycle();
			snrt_dma_start_2d(local_a_C02 + 3 * 256, *local_a_C01_C02 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_A03_from_C01_end = mcycle();
		}
	}


	// C03_load_A02_from_C02
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C03_load_A02_from_C02_start = mcycle();
			snrt_dma_start_2d(local_a_C03 + 2 * 256, *local_a_C02_C03 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C03_load_A02_from_C02_end = mcycle();
		}
	}


	// C10_load_B00_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_B00_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C10 + 0 * 256, *local_b_C00_C10 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_B00_from_C00_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T6:


	// Cluster 00 Load the B20 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B20_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 2 * 256, *local_addr_B_C00 + 128, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B20_end = mcycle();
	    }
	}



	// C03_load_A03_from_C02
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C03_load_A03_from_C02_start = mcycle();
			snrt_dma_start_2d(local_a_C03 + 3 * 256, *local_a_C02_C03 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C03_load_A03_from_C02_end = mcycle();
		}
	}


	// C10_load_B10_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_B10_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C10 + 1 * 256, *local_b_C00_C10 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_B10_from_C00_end = mcycle();
		}
	}


	// C20_load_B00_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_B00_from_C10_start = mcycle();
			snrt_dma_start_2d(local_b_C20 + 0 * 256, *local_b_C10_C20 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_B00_from_C10_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T7:


	// Cluster 00 Load the B30 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B30_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 3 * 256, *local_addr_B_C00 + 192, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B30_end = mcycle();
	    }
	}



	// C10_load_B20_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_B20_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C10 + 2 * 256, *local_b_C00_C10 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_B20_from_C00_end = mcycle();
		}
	}


	// C20_load_B10_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_B10_from_C10_start = mcycle();
			snrt_dma_start_2d(local_b_C20 + 1 * 256, *local_b_C10_C20 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_B10_from_C10_end = mcycle();
		}
	}


	// C30_load_B00_from_C20
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C30_load_B00_from_C20_start = mcycle();
			snrt_dma_start_2d(local_b_C30 + 0 * 256, *local_b_C20_C30 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C30_load_B00_from_C20_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T8:


	// Cluster 00 Load the A10 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A10_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 0 * 256, *local_addr_A_C00 + 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A10_end = mcycle();
	    }
	}



	// C10_load_B30_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_B30_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C10 + 3 * 256, *local_b_C00_C10 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_B30_from_C00_end = mcycle();
		}
	}


	// C20_load_B20_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_B20_from_C10_start = mcycle();
			snrt_dma_start_2d(local_b_C20 + 2 * 256, *local_b_C10_C20 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_B20_from_C10_end = mcycle();
		}
	}


	// C30_load_B10_from_C20
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C30_load_B10_from_C20_start = mcycle();
			snrt_dma_start_2d(local_b_C30 + 1 * 256, *local_b_C20_C30 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C30_load_B10_from_C20_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T9:


	// Cluster 00 Load the A11 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A11_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 1 * 256, *local_addr_A_C00 + 320, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A11_end = mcycle();
	    }
	}



	// C10_load_A10_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A10_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 0 * 256, *local_a_C00_C10 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A10_from_C00_end = mcycle();
		}
	}


	// C20_load_B30_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_B30_from_C10_start = mcycle();
			snrt_dma_start_2d(local_b_C20 + 3 * 256, *local_b_C10_C20 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_B30_from_C10_end = mcycle();
		}
	}


	// C30_load_B20_from_C20
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C30_load_B20_from_C20_start = mcycle();
			snrt_dma_start_2d(local_b_C30 + 2 * 256, *local_b_C20_C30 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C30_load_B20_from_C20_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T10:


	// Cluster 00 Load the A12 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A12_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 2 * 256, *local_addr_A_C00 + 384, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A12_end = mcycle();
	    }
	}



	// C10_load_A11_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A11_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 1 * 256, *local_a_C00_C10 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A11_from_C00_end = mcycle();
		}
	}


	// C11_load_A10_from_C10
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C11_load_A10_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C11 + 0 * 256, *local_a_C10_C11 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C11_load_A10_from_C10_end = mcycle();
		}
	}


	// C30_load_B30_from_C20
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C30_load_B30_from_C20_start = mcycle();
			snrt_dma_start_2d(local_b_C30 + 3 * 256, *local_b_C20_C30 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C30_load_B30_from_C20_end = mcycle();
		}
	}
	//T11:


	// Cluster 00 Load the A13 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A13_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 3 * 256, *local_addr_A_C00 + 448, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A13_end = mcycle();
	    }
	}



	// C10_load_A12_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A12_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 2 * 256, *local_a_C00_C10 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A12_from_C00_end = mcycle();
		}
	}


	// C11_load_A11_from_C10
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C11_load_A11_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C11 + 1 * 256, *local_a_C10_C11 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C11_load_A11_from_C10_end = mcycle();
		}
	}


	// C12_load_A10_from_C11
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C12_load_A10_from_C11_start = mcycle();
			snrt_dma_start_2d(local_a_C12 + 0 * 256, *local_a_C11_C12 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C12_load_A10_from_C11_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T12:


	// Cluster 00 Load the B01 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B01_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 0 * 256, *local_addr_B_C00 + 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B01_end = mcycle();
	    }
	}



	// C10_load_A13_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A13_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 3 * 256, *local_a_C00_C10 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A13_from_C00_end = mcycle();
		}
	}


	// C11_load_A12_from_C10
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C11_load_A12_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C11 + 2 * 256, *local_a_C10_C11 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C11_load_A12_from_C10_end = mcycle();
		}
	}


	// C12_load_A11_from_C11
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C12_load_A11_from_C11_start = mcycle();
			snrt_dma_start_2d(local_a_C12 + 1 * 256, *local_a_C11_C12 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C12_load_A11_from_C11_end = mcycle();
		}
	}


	// C13_load_A10_from_C12
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C13_load_A10_from_C12_start = mcycle();
			snrt_dma_start_2d(local_a_C13 + 0 * 256, *local_a_C12_C13 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C13_load_A10_from_C12_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T13:


	// Cluster 00 Load the B11 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B11_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 1 * 256, *local_addr_B_C00 + 320, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B11_end = mcycle();
	    }
	}



	// C01_load_B01_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B01_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 0 * 256, *local_b_C00_C01 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B01_from_C00_end = mcycle();
		}
	}


	// C11_load_A13_from_C10
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C11_load_A13_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C11 + 3 * 256, *local_a_C10_C11 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C11_load_A13_from_C10_end = mcycle();
		}
	}


	// C12_load_A12_from_C11
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C12_load_A12_from_C11_start = mcycle();
			snrt_dma_start_2d(local_a_C12 + 2 * 256, *local_a_C11_C12 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C12_load_A12_from_C11_end = mcycle();
		}
	}


	// C13_load_A11_from_C12
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C13_load_A11_from_C12_start = mcycle();
			snrt_dma_start_2d(local_a_C13 + 1 * 256, *local_a_C12_C13 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C13_load_A11_from_C12_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T14:


	// Cluster 00 Load the B21 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B21_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 2 * 256, *local_addr_B_C00 + 384, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B21_end = mcycle();
	    }
	}



	// C01_load_B11_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B11_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 1 * 256, *local_b_C00_C01 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B11_from_C00_end = mcycle();
		}
	}


	// C11_load_B01_from_C01
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C11_load_B01_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C11 + 0 * 256, *local_b_C01_C11 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C11_load_B01_from_C01_end = mcycle();
		}
	}


	// C12_load_A13_from_C11
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C12_load_A13_from_C11_start = mcycle();
			snrt_dma_start_2d(local_a_C12 + 3 * 256, *local_a_C11_C12 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C12_load_A13_from_C11_end = mcycle();
		}
	}


	// C13_load_A12_from_C12
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C13_load_A12_from_C12_start = mcycle();
			snrt_dma_start_2d(local_a_C13 + 2 * 256, *local_a_C12_C13 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C13_load_A12_from_C12_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T15:


	// Cluster 00 Load the B31 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B31_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 3 * 256, *local_addr_B_C00 + 448, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B31_end = mcycle();
	    }
	}



	// C01_load_B21_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B21_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 2 * 256, *local_b_C00_C01 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B21_from_C00_end = mcycle();
		}
	}


	// C11_load_B11_from_C01
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C11_load_B11_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C11 + 1 * 256, *local_b_C01_C11 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C11_load_B11_from_C01_end = mcycle();
		}
	}


	// C13_load_A13_from_C12
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C13_load_A13_from_C12_start = mcycle();
			snrt_dma_start_2d(local_a_C13 + 3 * 256, *local_a_C12_C13 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C13_load_A13_from_C12_end = mcycle();
		}
	}


	// C21_load_B01_from_C11
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C21_load_B01_from_C11_start = mcycle();
			snrt_dma_start_2d(local_b_C21 + 0 * 256, *local_b_C11_C21 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C21_load_B01_from_C11_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T16:


	// Cluster 00 Load the A20 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A20_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 0 * 256, *local_addr_A_C00 + 512, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A20_end = mcycle();
	    }
	}



	// C01_load_B31_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B31_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 3 * 256, *local_b_C00_C01 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B31_from_C00_end = mcycle();
		}
	}


	// C11_load_B21_from_C01
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C11_load_B21_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C11 + 2 * 256, *local_b_C01_C11 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C11_load_B21_from_C01_end = mcycle();
		}
	}


	// C21_load_B11_from_C11
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C21_load_B11_from_C11_start = mcycle();
			snrt_dma_start_2d(local_b_C21 + 1 * 256, *local_b_C11_C21 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C21_load_B11_from_C11_end = mcycle();
		}
	}


	// C31_load_B01_from_C21
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C31_load_B01_from_C21_start = mcycle();
			snrt_dma_start_2d(local_b_C31 + 0 * 256, *local_b_C21_C31 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C31_load_B01_from_C21_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T17:


	// Cluster 00 Load the A21 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A21_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 1 * 256, *local_addr_A_C00 + 576, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A21_end = mcycle();
	    }
	}



	// C10_load_A20_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A20_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 0 * 256, *local_a_C00_C10 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A20_from_C00_end = mcycle();
		}
	}


	// C11_load_B31_from_C01
	if((snrt_global_core_idx()>=group5_bound_lower) && (snrt_global_core_idx()<group5_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C11_load_B31_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C11 + 3 * 256, *local_b_C01_C11 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C11_load_B31_from_C01_end = mcycle();
		}
	}


	// C21_load_B21_from_C11
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C21_load_B21_from_C11_start = mcycle();
			snrt_dma_start_2d(local_b_C21 + 2 * 256, *local_b_C11_C21 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C21_load_B21_from_C11_end = mcycle();
		}
	}


	// C31_load_B11_from_C21
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C31_load_B11_from_C21_start = mcycle();
			snrt_dma_start_2d(local_b_C31 + 1 * 256, *local_b_C21_C31 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C31_load_B11_from_C21_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T18:


	// Cluster 00 Load the A22 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A22_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 2 * 256, *local_addr_A_C00 + 640, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A22_end = mcycle();
	    }
	}



	// C10_load_A21_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A21_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 1 * 256, *local_a_C00_C10 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A21_from_C00_end = mcycle();
		}
	}


	// C20_load_A20_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_A20_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C20 + 0 * 256, *local_a_C10_C20 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_A20_from_C10_end = mcycle();
		}
	}


	// C21_load_B31_from_C11
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C21_load_B31_from_C11_start = mcycle();
			snrt_dma_start_2d(local_b_C21 + 3 * 256, *local_b_C11_C21 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C21_load_B31_from_C11_end = mcycle();
		}
	}


	// C31_load_B21_from_C21
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C31_load_B21_from_C21_start = mcycle();
			snrt_dma_start_2d(local_b_C31 + 2 * 256, *local_b_C21_C31 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C31_load_B21_from_C21_end = mcycle();
		}
	}
	//T19:


	// Cluster 00 Load the A23 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A23_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 3 * 256, *local_addr_A_C00 + 704, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A23_end = mcycle();
	    }
	}



	// C10_load_A22_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A22_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 2 * 256, *local_a_C00_C10 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A22_from_C00_end = mcycle();
		}
	}


	// C20_load_A21_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_A21_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C20 + 1 * 256, *local_a_C10_C20 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_A21_from_C10_end = mcycle();
		}
	}


	// C21_load_A20_from_C20
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C21_load_A20_from_C20_start = mcycle();
			snrt_dma_start_2d(local_a_C21 + 0 * 256, *local_a_C20_C21 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C21_load_A20_from_C20_end = mcycle();
		}
	}


	// C31_load_B31_from_C21
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C31_load_B31_from_C21_start = mcycle();
			snrt_dma_start_2d(local_b_C31 + 3 * 256, *local_b_C21_C31 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C31_load_B31_from_C21_end = mcycle();
		}
	}
	//T20:


	// Cluster 00 Load the B02 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B02_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 0 * 256, *local_addr_B_C00 + 512, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B02_end = mcycle();
	    }
	}



	// C10_load_A23_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A23_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 3 * 256, *local_a_C00_C10 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A23_from_C00_end = mcycle();
		}
	}


	// C20_load_A22_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_A22_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C20 + 2 * 256, *local_a_C10_C20 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_A22_from_C10_end = mcycle();
		}
	}


	// C21_load_A21_from_C20
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C21_load_A21_from_C20_start = mcycle();
			snrt_dma_start_2d(local_a_C21 + 1 * 256, *local_a_C20_C21 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C21_load_A21_from_C20_end = mcycle();
		}
	}


	// C22_load_A20_from_C21
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C22_load_A20_from_C21_start = mcycle();
			snrt_dma_start_2d(local_a_C22 + 0 * 256, *local_a_C21_C22 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C22_load_A20_from_C21_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T21:


	// Cluster 00 Load the B12 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B12_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 1 * 256, *local_addr_B_C00 + 576, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B12_end = mcycle();
	    }
	}



	// C01_load_B02_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B02_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 0 * 256, *local_b_C00_C01 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B02_from_C00_end = mcycle();
		}
	}


	// C20_load_A23_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_A23_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C20 + 3 * 256, *local_a_C10_C20 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_A23_from_C10_end = mcycle();
		}
	}


	// C21_load_A22_from_C20
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C21_load_A22_from_C20_start = mcycle();
			snrt_dma_start_2d(local_a_C21 + 2 * 256, *local_a_C20_C21 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C21_load_A22_from_C20_end = mcycle();
		}
	}


	// C22_load_A21_from_C21
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C22_load_A21_from_C21_start = mcycle();
			snrt_dma_start_2d(local_a_C22 + 1 * 256, *local_a_C21_C22 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C22_load_A21_from_C21_end = mcycle();
		}
	}


	// C23_load_A20_from_C22
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C23_load_A20_from_C22_start = mcycle();
			snrt_dma_start_2d(local_a_C23 + 0 * 256, *local_a_C22_C23 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C23_load_A20_from_C22_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T22:


	// Cluster 00 Load the B22 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B22_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 2 * 256, *local_addr_B_C00 + 640, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B22_end = mcycle();
	    }
	}



	// C01_load_B12_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B12_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 1 * 256, *local_b_C00_C01 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B12_from_C00_end = mcycle();
		}
	}


	// C02_load_B02_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_B02_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C02 + 0 * 256, *local_b_C01_C02 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_B02_from_C01_end = mcycle();
		}
	}


	// C21_load_A23_from_C20
	if((snrt_global_core_idx()>=group9_bound_lower) && (snrt_global_core_idx()<group9_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C21_load_A23_from_C20_start = mcycle();
			snrt_dma_start_2d(local_a_C21 + 3 * 256, *local_a_C20_C21 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C21_load_A23_from_C20_end = mcycle();
		}
	}


	// C22_load_A22_from_C21
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C22_load_A22_from_C21_start = mcycle();
			snrt_dma_start_2d(local_a_C22 + 2 * 256, *local_a_C21_C22 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C22_load_A22_from_C21_end = mcycle();
		}
	}


	// C23_load_A21_from_C22
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C23_load_A21_from_C22_start = mcycle();
			snrt_dma_start_2d(local_a_C23 + 1 * 256, *local_a_C22_C23 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C23_load_A21_from_C22_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T23:


	// Cluster 00 Load the B32 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B32_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 3 * 256, *local_addr_B_C00 + 704, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B32_end = mcycle();
	    }
	}



	// C01_load_B22_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B22_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 2 * 256, *local_b_C00_C01 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B22_from_C00_end = mcycle();
		}
	}


	// C02_load_B12_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_B12_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C02 + 1 * 256, *local_b_C01_C02 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_B12_from_C01_end = mcycle();
		}
	}


	// C12_load_B02_from_C02
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C12_load_B02_from_C02_start = mcycle();
			snrt_dma_start_2d(local_b_C12 + 0 * 256, *local_b_C02_C12 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C12_load_B02_from_C02_end = mcycle();
		}
	}


	// C22_load_A23_from_C21
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C22_load_A23_from_C21_start = mcycle();
			snrt_dma_start_2d(local_a_C22 + 3 * 256, *local_a_C21_C22 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C22_load_A23_from_C21_end = mcycle();
		}
	}


	// C23_load_A22_from_C22
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C23_load_A22_from_C22_start = mcycle();
			snrt_dma_start_2d(local_a_C23 + 2 * 256, *local_a_C22_C23 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C23_load_A22_from_C22_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T24:


	// Cluster 00 Load the A30 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A30_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 0 * 256, *local_addr_A_C00 + 768, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A30_end = mcycle();
	    }
	}



	// C01_load_B32_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B32_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 3 * 256, *local_b_C00_C01 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B32_from_C00_end = mcycle();
		}
	}


	// C02_load_B22_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_B22_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C02 + 2 * 256, *local_b_C01_C02 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_B22_from_C01_end = mcycle();
		}
	}


	// C12_load_B12_from_C02
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C12_load_B12_from_C02_start = mcycle();
			snrt_dma_start_2d(local_b_C12 + 1 * 256, *local_b_C02_C12 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C12_load_B12_from_C02_end = mcycle();
		}
	}


	// C22_load_B02_from_C12
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C22_load_B02_from_C12_start = mcycle();
			snrt_dma_start_2d(local_b_C22 + 0 * 256, *local_b_C12_C22 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C22_load_B02_from_C12_end = mcycle();
		}
	}


	// C23_load_A23_from_C22
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C23_load_A23_from_C22_start = mcycle();
			snrt_dma_start_2d(local_a_C23 + 3 * 256, *local_a_C22_C23 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C23_load_A23_from_C22_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T25:


	// Cluster 00 Load the A31 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A31_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 1 * 256, *local_addr_A_C00 + 832, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A31_end = mcycle();
	    }
	}



	// C02_load_B32_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_B32_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C02 + 3 * 256, *local_b_C01_C02 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_B32_from_C01_end = mcycle();
		}
	}


	// C10_load_A30_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A30_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 0 * 256, *local_a_C00_C10 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A30_from_C00_end = mcycle();
		}
	}


	// C12_load_B22_from_C02
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C12_load_B22_from_C02_start = mcycle();
			snrt_dma_start_2d(local_b_C12 + 2 * 256, *local_b_C02_C12 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C12_load_B22_from_C02_end = mcycle();
		}
	}


	// C22_load_B12_from_C12
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C22_load_B12_from_C12_start = mcycle();
			snrt_dma_start_2d(local_b_C22 + 1 * 256, *local_b_C12_C22 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C22_load_B12_from_C12_end = mcycle();
		}
	}


	// C32_load_B02_from_C22
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C32_load_B02_from_C22_start = mcycle();
			snrt_dma_start_2d(local_b_C32 + 0 * 256, *local_b_C22_C32 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C32_load_B02_from_C22_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T26:


	// Cluster 00 Load the A32 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A32_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 2 * 256, *local_addr_A_C00 + 896, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A32_end = mcycle();
	    }
	}



	// C10_load_A31_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A31_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 1 * 256, *local_a_C00_C10 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A31_from_C00_end = mcycle();
		}
	}


	// C12_load_B32_from_C02
	if((snrt_global_core_idx()>=group6_bound_lower) && (snrt_global_core_idx()<group6_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C12_load_B32_from_C02_start = mcycle();
			snrt_dma_start_2d(local_b_C12 + 3 * 256, *local_b_C02_C12 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C12_load_B32_from_C02_end = mcycle();
		}
	}


	// C20_load_A30_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_A30_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C20 + 0 * 256, *local_a_C10_C20 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_A30_from_C10_end = mcycle();
		}
	}


	// C22_load_B22_from_C12
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C22_load_B22_from_C12_start = mcycle();
			snrt_dma_start_2d(local_b_C22 + 2 * 256, *local_b_C12_C22 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C22_load_B22_from_C12_end = mcycle();
		}
	}


	// C32_load_B12_from_C22
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C32_load_B12_from_C22_start = mcycle();
			snrt_dma_start_2d(local_b_C32 + 1 * 256, *local_b_C22_C32 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C32_load_B12_from_C22_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T27:


	// Cluster 00 Load the A33 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_A33_start = mcycle();
	    snrt_dma_start_2d(local_a_C00 + 3 * 256, *local_addr_A_C00 + 960, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_A33_end = mcycle();
	    }
	}



	// C10_load_A32_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A32_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 2 * 256, *local_a_C00_C10 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A32_from_C00_end = mcycle();
		}
	}


	// C20_load_A31_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_A31_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C20 + 1 * 256, *local_a_C10_C20 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_A31_from_C10_end = mcycle();
		}
	}


	// C22_load_B32_from_C12
	if((snrt_global_core_idx()>=group10_bound_lower) && (snrt_global_core_idx()<group10_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C22_load_B32_from_C12_start = mcycle();
			snrt_dma_start_2d(local_b_C22 + 3 * 256, *local_b_C12_C22 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C22_load_B32_from_C12_end = mcycle();
		}
	}


	// C30_load_A30_from_C20
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C30_load_A30_from_C20_start = mcycle();
			snrt_dma_start_2d(local_a_C30 + 0 * 256, *local_a_C20_C30 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C30_load_A30_from_C20_end = mcycle();
		}
	}


	// C32_load_B22_from_C22
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C32_load_B22_from_C22_start = mcycle();
			snrt_dma_start_2d(local_b_C32 + 2 * 256, *local_b_C22_C32 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C32_load_B22_from_C22_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T28:


	// Cluster 00 Load the B03 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B03_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 0 * 256, *local_addr_B_C00 + 768, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B03_end = mcycle();
	    }
	}



	// C10_load_A33_from_C00
	if((snrt_global_core_idx()>=group4_bound_lower) && (snrt_global_core_idx()<group4_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C10_load_A33_from_C00_start = mcycle();
			snrt_dma_start_2d(local_a_C10 + 3 * 256, *local_a_C00_C10 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C10_load_A33_from_C00_end = mcycle();
		}
	}


	// C20_load_A32_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_A32_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C20 + 2 * 256, *local_a_C10_C20 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_A32_from_C10_end = mcycle();
		}
	}


	// C30_load_A31_from_C20
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C30_load_A31_from_C20_start = mcycle();
			snrt_dma_start_2d(local_a_C30 + 1 * 256, *local_a_C20_C30 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C30_load_A31_from_C20_end = mcycle();
		}
	}


	// C31_load_A30_from_C30
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C31_load_A30_from_C30_start = mcycle();
			snrt_dma_start_2d(local_a_C31 + 0 * 256, *local_a_C30_C31 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C31_load_A30_from_C30_end = mcycle();
		}
	}


	// C32_load_B32_from_C22
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C32_load_B32_from_C22_start = mcycle();
			snrt_dma_start_2d(local_b_C32 + 3 * 256, *local_b_C22_C32 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C32_load_B32_from_C22_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T29:


	// Cluster 00 Load the B13 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B13_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 1 * 256, *local_addr_B_C00 + 832, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B13_end = mcycle();
	    }
	}



	// C01_load_B03_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B03_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 0 * 256, *local_b_C00_C01 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B03_from_C00_end = mcycle();
		}
	}


	// C20_load_A33_from_C10
	if((snrt_global_core_idx()>=group8_bound_lower) && (snrt_global_core_idx()<group8_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C20_load_A33_from_C10_start = mcycle();
			snrt_dma_start_2d(local_a_C20 + 3 * 256, *local_a_C10_C20 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C20_load_A33_from_C10_end = mcycle();
		}
	}


	// C30_load_A32_from_C20
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C30_load_A32_from_C20_start = mcycle();
			snrt_dma_start_2d(local_a_C30 + 2 * 256, *local_a_C20_C30 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C30_load_A32_from_C20_end = mcycle();
		}
	}


	// C31_load_A31_from_C30
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C31_load_A31_from_C30_start = mcycle();
			snrt_dma_start_2d(local_a_C31 + 1 * 256, *local_a_C30_C31 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C31_load_A31_from_C30_end = mcycle();
		}
	}


	// C32_load_A30_from_C31
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C32_load_A30_from_C31_start = mcycle();
			snrt_dma_start_2d(local_a_C32 + 0 * 256, *local_a_C31_C32 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C32_load_A30_from_C31_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T30:


	// Cluster 00 Load the B23 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B23_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 2 * 256, *local_addr_B_C00 + 896, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B23_end = mcycle();
	    }
	}



	// C01_load_B13_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B13_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 1 * 256, *local_b_C00_C01 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B13_from_C00_end = mcycle();
		}
	}


	// C02_load_B03_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_B03_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C02 + 0 * 256, *local_b_C01_C02 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_B03_from_C01_end = mcycle();
		}
	}


	// C30_load_A33_from_C20
	if((snrt_global_core_idx()>=group12_bound_lower) && (snrt_global_core_idx()<group12_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C30_load_A33_from_C20_start = mcycle();
			snrt_dma_start_2d(local_a_C30 + 3 * 256, *local_a_C20_C30 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C30_load_A33_from_C20_end = mcycle();
		}
	}


	// C31_load_A32_from_C30
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C31_load_A32_from_C30_start = mcycle();
			snrt_dma_start_2d(local_a_C31 + 2 * 256, *local_a_C30_C31 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C31_load_A32_from_C30_end = mcycle();
		}
	}


	// C32_load_A31_from_C31
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C32_load_A31_from_C31_start = mcycle();
			snrt_dma_start_2d(local_a_C32 + 1 * 256, *local_a_C31_C32 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C32_load_A31_from_C31_end = mcycle();
		}
	}


	// C33_load_A30_from_C32
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C33_load_A30_from_C32_start = mcycle();
			snrt_dma_start_2d(local_a_C33 + 0 * 256, *local_a_C32_C33 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C33_load_A30_from_C32_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T31:


	// Cluster 00 Load the B33 from L3
	if((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){
	  if (snrt_is_dm_core()) {
	     uint32_t dma_C00_load_B33_start = mcycle();
	    snrt_dma_start_2d(local_b_C00 + 3 * 256, *local_addr_B_C00 + 960, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
	    snrt_dma_wait_all();
	    uint32_t dma_C00_load_B33_end = mcycle();
	    }
	}



	// C01_load_B23_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B23_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 2 * 256, *local_b_C00_C01 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B23_from_C00_end = mcycle();
		}
	}


	// C02_load_B13_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_B13_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C02 + 1 * 256, *local_b_C01_C02 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_B13_from_C01_end = mcycle();
		}
	}


	// C03_load_B03_from_C02
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C03_load_B03_from_C02_start = mcycle();
			snrt_dma_start_2d(local_b_C03 + 0 * 256, *local_b_C02_C03 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C03_load_B03_from_C02_end = mcycle();
		}
	}


	// C31_load_A33_from_C30
	if((snrt_global_core_idx()>=group13_bound_lower) && (snrt_global_core_idx()<group13_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C31_load_A33_from_C30_start = mcycle();
			snrt_dma_start_2d(local_a_C31 + 3 * 256, *local_a_C30_C31 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C31_load_A33_from_C30_end = mcycle();
		}
	}


	// C32_load_A32_from_C31
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C32_load_A32_from_C31_start = mcycle();
			snrt_dma_start_2d(local_a_C32 + 2 * 256, *local_a_C31_C32 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C32_load_A32_from_C31_end = mcycle();
		}
	}


	// C33_load_A31_from_C32
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C33_load_A31_from_C32_start = mcycle();
			snrt_dma_start_2d(local_a_C33 + 1 * 256, *local_a_C32_C33 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C33_load_A31_from_C32_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T32:


	// C01_load_B33_from_C00
	if((snrt_global_core_idx()>=group1_bound_lower) && (snrt_global_core_idx()<group1_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C01_load_B33_from_C00_start = mcycle();
			snrt_dma_start_2d(local_b_C01 + 3 * 256, *local_b_C00_C01 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C01_load_B33_from_C00_end = mcycle();
		}
	}


	// C02_load_B23_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_B23_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C02 + 2 * 256, *local_b_C01_C02 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_B23_from_C01_end = mcycle();
		}
	}


	// C03_load_B13_from_C02
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C03_load_B13_from_C02_start = mcycle();
			snrt_dma_start_2d(local_b_C03 + 1 * 256, *local_b_C02_C03 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C03_load_B13_from_C02_end = mcycle();
		}
	}


	// C13_load_B03_from_C03
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C13_load_B03_from_C03_start = mcycle();
			snrt_dma_start_2d(local_b_C13 + 0 * 256, *local_b_C03_C13 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C13_load_B03_from_C03_end = mcycle();
		}
	}


	// C32_load_A33_from_C31
	if((snrt_global_core_idx()>=group14_bound_lower) && (snrt_global_core_idx()<group14_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C32_load_A33_from_C31_start = mcycle();
			snrt_dma_start_2d(local_a_C32 + 3 * 256, *local_a_C31_C32 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C32_load_A33_from_C31_end = mcycle();
		}
	}


	// C33_load_A32_from_C32
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C33_load_A32_from_C32_start = mcycle();
			snrt_dma_start_2d(local_a_C33 + 2 * 256, *local_a_C32_C33 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C33_load_A32_from_C32_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T33:


	// C02_load_B33_from_C01
	if((snrt_global_core_idx()>=group2_bound_lower) && (snrt_global_core_idx()<group2_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C02_load_B33_from_C01_start = mcycle();
			snrt_dma_start_2d(local_b_C02 + 3 * 256, *local_b_C01_C02 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C02_load_B33_from_C01_end = mcycle();
		}
	}


	// C03_load_B23_from_C02
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C03_load_B23_from_C02_start = mcycle();
			snrt_dma_start_2d(local_b_C03 + 2 * 256, *local_b_C02_C03 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C03_load_B23_from_C02_end = mcycle();
		}
	}


	// C13_load_B13_from_C03
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C13_load_B13_from_C03_start = mcycle();
			snrt_dma_start_2d(local_b_C13 + 1 * 256, *local_b_C03_C13 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C13_load_B13_from_C03_end = mcycle();
		}
	}


	// C23_load_B03_from_C13
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C23_load_B03_from_C13_start = mcycle();
			snrt_dma_start_2d(local_b_C23 + 0 * 256, *local_b_C13_C23 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C23_load_B03_from_C13_end = mcycle();
		}
	}


	// C33_load_A33_from_C32
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C33_load_A33_from_C32_start = mcycle();
			snrt_dma_start_2d(local_a_C33 + 3 * 256, *local_a_C32_C33 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C33_load_A33_from_C32_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T34:


	// C03_load_B33_from_C02
	if((snrt_global_core_idx()>=group3_bound_lower) && (snrt_global_core_idx()<group3_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C03_load_B33_from_C02_start = mcycle();
			snrt_dma_start_2d(local_b_C03 + 3 * 256, *local_b_C02_C03 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C03_load_B33_from_C02_end = mcycle();
		}
	}


	// C13_load_B23_from_C03
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C13_load_B23_from_C03_start = mcycle();
			snrt_dma_start_2d(local_b_C13 + 2 * 256, *local_b_C03_C13 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C13_load_B23_from_C03_end = mcycle();
		}
	}


	// C23_load_B13_from_C13
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C23_load_B13_from_C13_start = mcycle();
			snrt_dma_start_2d(local_b_C23 + 1 * 256, *local_b_C13_C23 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C23_load_B13_from_C13_end = mcycle();
		}
	}


	// C33_load_B03_from_C23
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C33_load_B03_from_C23_start = mcycle();
			snrt_dma_start_2d(local_b_C33 + 0 * 256, *local_b_C23_C33 + 0 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C33_load_B03_from_C23_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T35:


	// C13_load_B33_from_C03
	if((snrt_global_core_idx()>=group7_bound_lower) && (snrt_global_core_idx()<group7_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C13_load_B33_from_C03_start = mcycle();
			snrt_dma_start_2d(local_b_C13 + 3 * 256, *local_b_C03_C13 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C13_load_B33_from_C03_end = mcycle();
		}
	}


	// C23_load_B23_from_C13
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C23_load_B23_from_C13_start = mcycle();
			snrt_dma_start_2d(local_b_C23 + 2 * 256, *local_b_C13_C23 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C23_load_B23_from_C13_end = mcycle();
		}
	}


	// C33_load_B13_from_C23
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C33_load_B13_from_C23_start = mcycle();
			snrt_dma_start_2d(local_b_C33 + 1 * 256, *local_b_C23_C33 + 1 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C33_load_B13_from_C23_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T36:


	// C23_load_B33_from_C13
	if((snrt_global_core_idx()>=group11_bound_lower) && (snrt_global_core_idx()<group11_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C23_load_B33_from_C13_start = mcycle();
			snrt_dma_start_2d(local_b_C23 + 3 * 256, *local_b_C13_C23 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C23_load_B33_from_C13_end = mcycle();
		}
	}


	// C33_load_B23_from_C23
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C33_load_B23_from_C23_start = mcycle();
			snrt_dma_start_2d(local_b_C33 + 2 * 256, *local_b_C23_C33 + 2 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C33_load_B23_from_C23_end = mcycle();
		}
	}

	snrt_global_barrier();
	// ******************************************************************//
	//T37:


	// C33_load_B33_from_C23
	if((snrt_global_core_idx()>=group15_bound_lower) && (snrt_global_core_idx()<group15_bound_upper)){
		if(snrt_is_dm_core()){
			uint32_t dma_C33_load_B33_from_C23_start = mcycle();
			snrt_dma_start_2d(local_b_C33 + 3 * 256, *local_b_C23_C33 + 3 * 256, 8 * 8 * sizeof(int8_t), 1024, 1024, 8);
			snrt_dma_wait_all();
			uint32_t dma_C33_load_B33_from_C23_end = mcycle();
		}
	}

	// ******************************************************************//
	//                           END THE GEMM                            //
	// ******************************************************************//

	//return to host
	return_to_cva6(SYNC_ALL);

}

