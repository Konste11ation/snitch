#!/usr/bin/env python3

# Copyright 2023 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os





def get_addr_map(**kwargs):
    strideInnermostA = kwargs["strideInnermostA"]
    ldA = kwargs["ldA"]
    strideA = kwargs["strideA"]

    strideInnermostB = kwargs["strideInnermostB"]
    ldB = kwargs["ldB"]
    strideB = kwargs["strideB"]

    addr_map_C0 = {'A00':  'C0_ADDR_a',
                   'B00':  'C0_ADDR_b'}
    addr_map_C1 = {'A00':  'C1_ADDR_a',
                   'B00':  'C1_ADDR_b'}
    addr_map_C2 = {'A00':  'C2_ADDR_a',
                   'B00':  'C2_ADDR_b'}
    addr_map_C3 = {'A00':  'C3_ADDR_a',
                   'B00':  'C3_ADDR_b'}
    addr_map = {
        'C0': addr_map_C0,
        'C1': addr_map_C1,
        'C2': addr_map_C2,
        'C3': addr_map_C3
    }
    return addr_map

def get_result_map():
    result_map = {
        'C0': "C00",
        'C1': "C01",
        'C2': "C10",
        'C3': "C11"
    }
    return result_map

def emit_data_flow_table():
    code_string = "\n"
    code_string = """
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
    """
    return code_string

def A_idx_2_gemm_idx(A_idx_x,A_idx_y):
    gemm_idx_m = A_idx_x
    gemm_idx_k = A_idx_y
    return gemm_idx_m , gemm_idx_k

def B_idx_2_gemm_idx(B_idx_x,B_idx_y):
    gemm_idx_n = B_idx_y
    gemm_idx_k = B_idx_x
    return gemm_idx_n , gemm_idx_k

def get_delta_addr_A(m,k,**kwargs):

    delta_addr_A = m * kwargs["meshRow"] * kwargs["tileSize"] * kwargs["K"] + \
                   k * kwargs["meshRow"] * kwargs["tileSize"];  
    return delta_addr_A

def get_delta_addr_a(m,k,**kwargs):
    delta_addr_a= m * kwargs["ldA"] + k * kwargs["strideInnermostA"]
    return delta_addr_a

def get_delta_addr_B(n,k,**kwargs):

    delta_addr_B = n * kwargs["tileSize"] * kwargs["meshCol"] * kwargs["K"] + \
                   k * kwargs["tileSize"] * kwargs["meshCol"];  
    return delta_addr_B

def get_delta_addr_b(n,k,**kwargs):
    delta_addr_b =  n * kwargs["ldB"] + k * kwargs["strideInnermostB"]
    return delta_addr_b

def set_batch_gemm(cluster_num,size_setting,subtraction_setting,**kwargs):
    strideInnermostA = kwargs["strideInnermostA"]
    strideInnermostB = kwargs["strideInnermostB"]
    strideInnermostC = kwargs["strideInnermostC"]

    ldA = kwargs["ldA"]
    ldB = kwargs["ldB"]
    ldC = kwargs["ldC"]

    strideA = kwargs["strideA"]
    strideB = kwargs["strideB"]
    strideC = kwargs["strideC"]
    code_string =  "\t// Set matrix size\n"
    code_string += f"\twrite_csr(0x3c0, {size_setting});\n"
    code_string +=  "\n"
    code_string +=  "\t// Set addresses\n"
    code_string += f"\twrite_csr(0x3c1, (uint32_t)local_a_{cluster_num});\n"
    code_string += f"\twrite_csr(0x3c2, (uint32_t)local_b_{cluster_num});\n"
    code_string += f"\twrite_csr(0x3c3, (uint32_t)local_c_{cluster_num});\n"
    code_string +=  "\n"
    code_string +=  "\t// Set Strides\n"
    code_string += f"\twrite_csr(0x3c4, {strideInnermostA});\n"
    code_string += f"\twrite_csr(0x3c5, {strideInnermostB});\n"
    code_string += f"\twrite_csr(0x3c6, {strideInnermostC});\n"
    code_string +=  "\n"
    code_string += f"\twrite_csr(0x3c7, {ldA});\n"
    code_string += f"\twrite_csr(0x3c8, {ldB});\n"
    code_string += f"\twrite_csr(0x3c9, {ldC});\n"
    code_string +=  "\n"
    code_string += f"\twrite_csr(0x3ca, {strideA});\n"
    code_string += f"\twrite_csr(0x3cb, {strideB});\n"
    code_string += f"\twrite_csr(0x3cc, {strideC});\n"
    code_string +=  "\n"
    code_string +=  "\t// Set subtraction values\n"
    code_string += f"\twrite_csr(0x3ce, {subtraction_setting});\n"
    return code_string

def get_size_config(**kwargs):
    Batch = kwargs["Batch"]
    M = kwargs["M_cluster"]
    K = kwargs["K_cluster"]
    N = kwargs["N_cluster"]
    code_string = f"((uint32_t){Batch} << 24) | ((uint32_t){M} << 16) | ((uint32_t){K} << 8) | ((uint32_t){N}) "
    return code_string

def get_substraction_config(**kwargs):
    code_string = f"(int32_t)(((uint8_t)(51) << 8) | (uint8_t)(-26))"
    return code_string







def emit_check_result_function():
    check_result = "\n"
    check_result += """
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


"""
    return check_result

def emit_global_ADDR(**kwargs):
    global_addr = "\n"
    for i in range(kwargs["NUM_CLUSTER"]):
        global_addr += f"""
uint8_t*  C{i}_ADDR_a;
uint8_t*  C{i}_ADDR_b;
uint32_t* C{i}_ADDR_c;
"""
    return global_addr

def emit_main_header():
    main_string = "\n"
    main_string += "int main() {\n"
    return main_string

def emit_main_end():
    main_string = "\n"
    main_string += "}\n"
    return main_string
def emit_post_wakeup_cl():
    code_post_wakeup = "\n"
    code_post_wakeup += "\tpost_wakeup_cl();\n"
    return code_post_wakeup

def emit_group_idx_comment(**kwargs):
    group_idx_comment = "\n"
    group_idx_comment += "\t// Core            hard_id   Global Core idx = hartid - base_hardid = hartid - 1\n"
    group_idx_comment += "\t// ---------------------------\n"
    group_idx_comment += "\t// CVA6            0\n"
    group_idx_comment += "\t// ---------------------------\n"
    for i in range(kwargs["NUM_CLUSTER"]):
        group_idx_comment += f"\t// Group{i}:SNAX     {2*i+1}         {2*i}\n"
        group_idx_comment += f"\t// Group{i}:SNAX     {2*i+2}         {2*i+1}\n"
        group_idx_comment += "\t// ---------------------------\n"
    return group_idx_comment

def emit_group_boundary(**kwargs):
    code_string = "\n"
    for i in range(kwargs["NUM_CLUSTER"]):
        if(i==0):
            code_string+= "\tuint32_t group0_bound_lower = 0;\n"
            code_string+= "\tuint32_t group0_bound_upper = snrt_cluster_core_num();\n"
        else:
            code_string+= f"\tuint32_t group{i}_bound_lower = group{i-1}_bound_upper;\n"
            code_string+= f"\tuint32_t group{i}_bound_upper = {i+1}*snrt_cluster_core_num();\n"
    return code_string

def emit_size_setting(**kwargs):
    code_string = "\n"
    M_cluster = kwargs["M_cluster"]
    K_cluster = kwargs["K_cluster"]
    N_cluster = kwargs["N_cluster"]
    Batch = kwargs["Batch"]
    code_string += f"\tuint32_t size_setting = gen_size_config((uint8_t){Batch}, (uint8_t){M_cluster}, (uint8_t){K_cluster}, (uint8_t){N_cluster});\n"
    return code_string

def emit_subtraction_setting():
    code_string = "\n"
    code_string += "\tuint32_t subtraction_setting = gen_subtraction_config(subtraction_a, subtraction_b);\n"
    return code_string

def emit_init_C0_TCDM(**kwargs):
    delta_local_a = kwargs["delta_local_a"]
    delta_local_b = kwargs["delta_local_b"]
    code_string = "\n"
    code_string += "\tif((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper))"
    code_string += "{\n"
    code_string += "\t\t// Start a new row to store err\n"
    code_string += "\t\tlocal_err_0 = (int32_t*)snrt_l1_next();\n"
    code_string += "\t\tlocal_addr_A_0 = (int8_t**)(local_err_0 + 1);\n"
    code_string += "\t\tlocal_addr_B_0 = (int8_t**)(local_addr_A_0 + 4);\n"
    code_string += "\n"
    code_string += "\t\t// Space to store matrix\n"
    code_string += "\t\tlocal_a_0 = (int8_t*)(local_err_0 + 256 * sizeof(int8_t));\n"
    code_string += f"\t\tlocal_b_0 = local_a_0 + {delta_local_a}* sizeof(int8_t);\n"
    code_string += f"\t\tlocal_c_0 = (int32_t*)(local_b_0 + {delta_local_b} * sizeof(int8_t));\n"
    code_string += "\n"
    code_string += "\t\tC0_ADDR_a = local_a_0;\n"
    code_string += "\t\tC0_ADDR_b = local_b_0;\n"
    code_string += "\t\tC0_ADDR_c = local_c_0;\n"
    code_string += "\n"
    code_string += "\t\t*local_err_0 = 0;\n"
    code_string += "\t\t*local_addr_A_0 = I0;\n"
    code_string += "\t\t*local_addr_B_0 = W0;\n"
    code_string += "\t}\n"
    return code_string


def emit_init_TCDM_addr(cluster_num,**kwargs):
    code_string = "\n"

    if(cluster_num==0):
        code_string += "\t// TCDM Memory for Cluster0\n"
        code_string += "\tint32_t* local_err_0;\n"
        code_string += "\tint8_t** local_addr_A_0;\n"
        code_string += "\tint8_t** local_addr_B_0;\n"        
        code_string += "\tint8_t*  local_a_0;\n"
        code_string += "\tint8_t*  local_b_0;\n"
        code_string += "\tint32_t* local_c_0;\n"
    else:
        cluster_idx_list = [j for j in range(kwargs["NUM_CLUSTER"]) if j!=cluster_num]
        code_string += f"\t// TCDM Memory for Cluster{cluster_num}\n"
        code_string += f"\tint32_t* local_err_{cluster_num};\n"
        code_string += f"\tint8_t*  local_a_{cluster_num};\n"
        code_string += f"\tint8_t*  local_b_{cluster_num};\n"
        code_string += f"\tint32_t* local_c_{cluster_num};\n"
        for cluster_idx in cluster_idx_list:
            code_string += f"\tint8_t** local_a_C{cluster_idx}_{cluster_num};\n"
            code_string += f"\tint8_t** local_b_C{cluster_idx}_{cluster_num};\n"
            code_string += f"\tint32_t** local_c_C{cluster_idx}_{cluster_num};\n"
    return code_string


def emit_init_TCDM(**kwargs):
    delta_local_a = kwargs["delta_local_a"]
    delta_local_b = kwargs["delta_local_b"]
    code_string = "\n"
    code_string += "\t// Prepare addresses in TCDM for 4 cores\n"
    code_string += "\n"

    for i in range(kwargs["NUM_CLUSTER"]):
        code_string += emit_init_TCDM_addr(i,**kwargs)

    code_string += "\n"            

    for i in range(kwargs["NUM_CLUSTER"]):
        code_string += "\n"
        if(i==0):
            code_string += emit_init_C0_TCDM(**kwargs)
        else:
            code_string += f"\tif((snrt_global_core_idx()>=group{i}_bound_lower) && (snrt_global_core_idx()<group{i}_bound_upper))"
            code_string += "{\n"
            code_string += "\t\t// Start a new row to store err\n"
            code_string += f"\t\tlocal_err_{i} = (int32_t*)snrt_l1_next();\n"
            code_string += "\t\t// Space to store matrix\n"
            code_string += f"\t\tlocal_a_{i} = (int8_t*)(local_err_{i} + 256 * sizeof(int8_t));\n"
            code_string += f"\t\tlocal_b_{i} = local_a_{i} + {delta_local_a} * sizeof(int8_t);\n"
            code_string += f"\t\tlocal_c_{i} = (int32_t*)(local_b_{i} + {delta_local_b} * sizeof(int8_t));\n"
            cluster_idx_list = [j for j in range(kwargs["NUM_CLUSTER"]) if j!=i]
            loop_counter = 0
            for cluster_idx in cluster_idx_list:
                if(loop_counter==0):
                    code_string += f"\t\tlocal_a_C{cluster_idx}_{i} = (int8_t**)(local_err_{i} + 1);\n"
                else:
                    code_string += f"\t\tlocal_a_C{cluster_idx}_{i} = (int8_t**)({last_addr} + 4);\n"
                code_string += f"\t\tlocal_b_C{cluster_idx}_{i} = (int8_t**)(local_a_C{cluster_idx}_{i} + 4);\n"
                last_addr =  f"local_b_C{cluster_idx}_{i}"
                loop_counter +=1
            code_string += "\n"
            code_string += f"\t\tC{i}_ADDR_a = local_a_{i};\n"
            code_string += f"\t\tC{i}_ADDR_b = local_b_{i};\n"
            code_string += f"\t\tC{i}_ADDR_c = local_c_{i};\n"
            code_string += "\n"
            code_string += f"\t\t*local_err_{i} = 0;\n"
            code_string += "\t}\n"
    return code_string

def emit_broadcast_TCDM_addr(**kwargs):
    code_string = "\n"
    for i in range(kwargs["NUM_CLUSTER"]):
        if(i==0):
            code_string += "\t//Cluster0 already got the matrix A and B addr\n"
            code_string += "\n"
        else:
            cluster_idx_list = [j for j in range(kwargs["NUM_CLUSTER"]) if j!=i]
            code_string += "\n"
            cluster_idx_list_unpack = ' '.join([str(item) for item in cluster_idx_list])
            code_string += f"\t//Cluster{i} need the addr of a and b for Cluster{cluster_idx_list_unpack}\n"
            code_string += f"\tif((snrt_global_core_idx()>=group{i}_bound_lower) && (snrt_global_core_idx()<group{i}_bound_upper))"
            code_string +=  "{\n"
            for cluster_idx in cluster_idx_list:
                code_string += f"\t  *local_a_C{cluster_idx}_{i} = C{cluster_idx}_ADDR_a;\n"
                code_string += f"\t  *local_b_C{cluster_idx}_{i} = C{cluster_idx}_ADDR_b;\n"
            code_string +=  "\t}\n"         

    return code_string



def emit_global_barrier():
    global_barrier = "\n"
    global_barrier += "\tsnrt_global_barrier();\n"
    return global_barrier

def emit_start_gemm():
    start_gemm_comment = "\n"
    start_gemm_comment += "\t// ******************************************************************//\n"
    start_gemm_comment += "\t//                           START THE GEMM                          //\n"
    start_gemm_comment += "\t// ******************************************************************//\n"
    return start_gemm_comment

def emit_end_gemm():
    end_gemm_comment = "\n"
    end_gemm_comment += "\t// ******************************************************************//\n"
    end_gemm_comment += "\t//                           END THE GEMM                            //\n"
    end_gemm_comment += "\t// ******************************************************************//\n"
    return end_gemm_comment


def emit_C0_load_A_from_L3(A_idx_x,A_idx_y,**kwargs):
    Batch = kwargs["Batch"]
    strideA = kwargs["strideA"]
    meshRow = kwargs["meshRow"]
    tileSize = kwargs["tileSize"]
    gemm_idx_m, gemm_idx_k = A_idx_2_gemm_idx(A_idx_x,A_idx_y)

    delta_addr_a = get_delta_addr_a(gemm_idx_m,gemm_idx_k,**kwargs)
    delta_addr_A = get_delta_addr_A(gemm_idx_m,gemm_idx_k,**kwargs)
    stride_dst = strideA 
    stride_src = kwargs["M"] * kwargs["meshRow"] * kwargs["tileSize"] * kwargs["K"]


    matrix_str = f'A{A_idx_x}{A_idx_y}' 
    code_string = "\n"
    code_string += f"\t// Cluster 0 Load the {matrix_str} from L3\n"
    code_string +=  "\tif((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){\n"
    code_string +=  "\t  if (snrt_is_dm_core()) {\n"
    code_string += f"\t     uint32_t dma_C0_load_{matrix_str}_start = mcycle();\n"
    code_string += f"\t    snrt_dma_start_2d(local_a_0 + {delta_addr_a}, *local_addr_A_0 + {delta_addr_A}, {meshRow} * {tileSize} * sizeof(int8_t), {stride_dst}, {stride_src}, {Batch});\n"
    code_string += f"\t    snrt_dma_wait_all();\n"
    code_string += f"\t    uint32_t dma_C0_load_{matrix_str}_end = mcycle();\n"
    code_string +=  "\t    }\n"
    code_string +=  "\t}\n"
    code_string +=  "\n"
    return code_string

def emit_C0_load_B_from_L3(B_idx_x,B_idx_y,**kwargs):
    Batch = kwargs["Batch"]
    strideB = kwargs["strideB"]
    meshCol = kwargs["meshCol"]
    tileSize = kwargs["tileSize"]
    gemm_idx_n, gemm_idx_k = B_idx_2_gemm_idx(B_idx_x,B_idx_y)

    delta_addr_b = get_delta_addr_b(gemm_idx_n,gemm_idx_k,**kwargs)
    delta_addr_B = get_delta_addr_B(gemm_idx_n,gemm_idx_k,**kwargs)
    stride_dst = strideB
    stride_src = kwargs["K"] * kwargs["tileSize"] * kwargs["meshCol"] * kwargs["N"]


    matrix_str = f'B{B_idx_x}{B_idx_y}'    
    code_string = "\n"
    code_string += f"\t// Cluster 0 Load the {matrix_str} from L3\n"
    code_string +=  "\tif((snrt_global_core_idx()>=group0_bound_lower) && (snrt_global_core_idx()<group0_bound_upper)){\n"
    code_string +=  "\t  if (snrt_is_dm_core()) {\n"
    code_string += f"\t    uint32_t dma_C0_load_{matrix_str}_start = mcycle();\n"
    code_string += f"\t   snrt_dma_start_2d(local_b_0 + {delta_addr_b}, *local_addr_B_0 + {delta_addr_B}, {meshCol} * {tileSize} * sizeof(int8_t), {stride_dst}, {stride_src}, {Batch});\n"
    code_string += f"\t   snrt_dma_wait_all();\n"
    code_string += f"\t   uint32_t dma_C0_load_{matrix_str}_end = mcycle();\n"
    code_string +=  "\t   }\n"
    code_string +=  "\t}\n"
    code_string +=  "\n"
    return code_string


def emit_C0_load_from_L3(matrix,matrix_idx_x,matrix_idx_y,**kwargs):
    code_string = "\n"
    if(matrix=="A"):
        code_string += emit_C0_load_A_from_L3(matrix_idx_x,matrix_idx_y,**kwargs)
    if(matrix=="B"):
        code_string += emit_C0_load_B_from_L3(matrix_idx_x,matrix_idx_y,**kwargs)
    return code_string


def emit_init_loading_W(**kwargs):
    Batch = kwargs["Batch"]
    strideB = kwargs["strideB"]
    meshCol = kwargs["meshCol"]
    tileSize = kwargs["tileSize"]
    stride_dst = strideB
    stride_src = kwargs["K"] * kwargs["tileSize"] * kwargs["meshCol"] * kwargs["N"]
    
    code_string = "\n"
    for i in range(kwargs["NUM_CLUSTER"]):
        code_string += f"\t// Cluster {i} Load the W{i} from L3\n"
        code_string += f"\tif((snrt_global_core_idx()>=group{i}_bound_lower) && (snrt_global_core_idx()<group{i}_bound_upper))"
        code_string +=  "{\n"
        code_string +=  "\t  if (snrt_is_dm_core()) {\n"
        code_string += f"\t   snrt_dma_start_2d(local_b_{i}, W{i}, {meshCol} * {tileSize} * sizeof(int8_t), {stride_dst}, {stride_src}, {Batch});\n"
        code_string += f"\t   snrt_dma_wait_all();\n"
        code_string +=  "\t   }\n"
        code_string +=  "\t}\n"
        code_string +=  "\n"
    return code_string

def emit_Cx_load_A_from_Cy(init_cluster,target_cluster,A_idx_x,A_idx_y,addr_map,**kwargs):
    Batch = kwargs["Batch"]
    meshRow = kwargs["meshRow"]
    tileSize = kwargs["tileSize"]
    init_cluster_str = f'C{init_cluster}'
    target_cluster_str = f'C{target_cluster}'
    matrix_str = f'A{A_idx_x}{A_idx_y}'
    dma_src_addr = addr_map[target_cluster_str][matrix_str]
    dma_src_addr = dma_src_addr.replace(f"C{target_cluster}_ADDR_a",f"*local_a_C{target_cluster}_{init_cluster}")
    dma_dst_addr = addr_map[init_cluster_str][matrix_str]
    dma_dst_addr = dma_dst_addr.replace(f"C{init_cluster}_ADDR_a",f"local_a_{init_cluster}")
    strideA = kwargs["strideA"]
    code_string = "\n"
    code_string += f"\t// {init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}\n"
    code_string += f"\tif((snrt_global_core_idx()>=group{init_cluster}_bound_lower) && (snrt_global_core_idx()<group{init_cluster}_bound_upper))"
    code_string +=  "{\n"
    code_string +=  "\t\tif(snrt_is_dm_core()){\n"
    code_string += f"\t\t\tuint32_t dma_{init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}_start = mcycle();\n"
    code_string += f"\t\t\tsnrt_dma_start_2d({dma_dst_addr}, {dma_src_addr}, {meshRow} * {tileSize} * sizeof(int8_t), {strideA}, {strideA}, {Batch});\n"
    code_string +=  "\t\t\tsnrt_dma_wait_all();\n"
    code_string += f"\t\t\tuint32_t dma_{init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}_end = mcycle();\n"
    code_string +=  "\t\t}\n"
    code_string +=  "\t}\n"
    return code_string

def emit_Cx_load_B_from_Cy(init_cluster,target_cluster,B_idx_x,B_idx_y,addr_map,**kwargs):
    Batch = kwargs["Batch"]
    meshCol = kwargs["meshCol"]
    tileSize = kwargs["tileSize"]
    init_cluster_str = f'C{init_cluster}'
    target_cluster_str = f'C{target_cluster}'
    matrix_str = f'B{B_idx_x}{B_idx_y}'
    dma_src_addr = addr_map[target_cluster_str][matrix_str]
    dma_src_addr = dma_src_addr.replace(f"C{target_cluster}_ADDR_b",f"*local_b_C{target_cluster}_{init_cluster}")
    dma_dst_addr = addr_map[init_cluster_str][matrix_str]
    dma_dst_addr = dma_dst_addr.replace(f"C{init_cluster}_ADDR_b",f"local_b_{init_cluster}")
    strideB = kwargs["strideB"]
    code_string = "\n"
    code_string += f"\t// {init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}\n"
    code_string += f"\tif((snrt_global_core_idx()>=group{init_cluster}_bound_lower) && (snrt_global_core_idx()<group{init_cluster}_bound_upper))"
    code_string +=  "{\n"
    code_string +=  "\t\tif(snrt_is_dm_core()){\n"
    code_string += f"\t\t\tuint32_t dma_{init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}_start = mcycle();\n"
    code_string += f"\t\t\tsnrt_dma_start_2d({dma_dst_addr}, {dma_src_addr}, {meshCol} * {tileSize} * sizeof(int8_t), {strideB}, {strideB}, {Batch});\n"
    code_string +=  "\t\t\tsnrt_dma_wait_all();\n"
    code_string += f"\t\t\tuint32_t dma_{init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}_end = mcycle();\n"
    code_string +=  "\t\t}\n"
    code_string +=  "\t}\n"
    return code_string

def emit_Cx_load_from_Cy(init_cluster,target_cluster,matrix,matrix_idx_x,matrix_idx_y,addr_map,**kwargs):
    code_string = "\n"
    if(matrix=="A"):
        code_string += emit_Cx_load_A_from_Cy(init_cluster,target_cluster,matrix_idx_x,matrix_idx_y,addr_map,**kwargs)
    if(matrix=="B"):
        code_string += emit_Cx_load_B_from_Cy(init_cluster,target_cluster,matrix_idx_x,matrix_idx_y,addr_map,**kwargs)
    return code_string 

def emit_compute_gemm(cluster_num,result_map,**kwargs):
    cluster_str = f"C{cluster_num}"
    result_str = result_map[cluster_str]

    code_string = "\n"
    code_string += f"\t// Cluster{cluster_num} compute the {result_str}\n"
    code_string += f"\tif((snrt_global_core_idx()>=group{cluster_num}_bound_lower) && (snrt_global_core_idx()<group{cluster_num}_bound_upper))"
    code_string += "{\n"
    code_string +=  "\t\tif (snrt_is_compute_core()){\n"
    code_string += f"\t\t   uint32_t gemm_{result_str}_start = mcycle();\n"
    code_string +=  "\t\t   // Set GEMM configuration CSR \n"
    code_string +=  "\t\t   start_batch_gemm();\n"
    code_string +=  "\t\t   wait_batch_gemm();\n"
    code_string += f"\t\t   uint32_t gemm_{result_str}_end = mcycle();\n"
    code_string +=  "\t\t}\n"
    code_string +=  "\t}\n"
    return code_string

def emit_Cx_init_GeMM_setup(cluster_num,**kwargs):
    strideInnermostA = kwargs["strideInnermostA"]
    strideInnermostB = kwargs["strideInnermostB"]
    strideInnermostC = kwargs["strideInnermostC"]

    ldA = kwargs["ldA"]
    ldB = kwargs["ldB"]
    ldC = kwargs["ldC"]

    strideA = kwargs["strideA"]
    strideB = kwargs["strideB"]
    strideC = kwargs["strideC"]

    code_string = "\n"
    code_string += f"\t// Cluster{cluster_num} init the GeMM config\n"
    code_string += f"\tif((snrt_global_core_idx()>=group{cluster_num}_bound_lower) && (snrt_global_core_idx()<group{cluster_num}_bound_upper))"
    code_string += "{\n"
    code_string +=  "\t\tif (snrt_is_compute_core()){\n"
    code_string +=  "\t\t   // Set GEMM configuration CSR \n"
    code_string +=  "\t\t   set_batch_gemm(size_setting,\n"
    code_string += f"\t\t                  local_a_{cluster_num}, local_b_{cluster_num},\n"
    code_string +=  "\t\t                  subtraction_setting,\n"
    code_string += f"\t\t                  local_c_{cluster_num},\n"
    code_string += f"\t\t                  {strideInnermostA},\n"
    code_string += f"\t\t                  {strideInnermostB},\n"
    code_string += f"\t\t                  {strideInnermostC},\n"
    code_string += f"\t\t                  {ldA},\n"
    code_string += f"\t\t                  {ldB},\n"
    code_string += f"\t\t                  {ldC},\n"
    code_string += f"\t\t                  {strideA},\n"
    code_string += f"\t\t                  {strideB},\n"
    code_string += f"\t\t                  {strideC});\n"
    code_string +=  "\t\t}\n"
    code_string +=  "\t}\n"
    return code_string


def emit_init_GeMM_setup(**kwargs):
    code_string = "\n"
    for i in range(kwargs["NUM_CLUSTER"]):
        GeMM_setup_string = emit_Cx_init_GeMM_setup(i,**kwargs)
        code_string += GeMM_setup_string
        code_string += "\n"
    return code_string


def emit_code_seperate_comment():
    code_seperate_comment = "\t// ******************************************************************//\n"
    return code_seperate_comment

def emit_check_result(**kwargs):
    code_string = "\n"
    code_string += "\t//Check Results\n"
    code_string += "\t//Compare SNAX GEMM result with golden model\n"
    for i in range(kwargs["NUM_CLUSTER"]):
        code_string +=  "\n"
        code_string += f"\tif((snrt_global_core_idx()>=group{i}_bound_lower) && (snrt_global_core_idx()<group{i}_bound_upper))"
        code_string +=  "{\n"
        code_string +=  "\t\tif(snrt_is_compute_core()){\n"
        code_string += f"\t\t   *local_err_{i} += check_result_batch(Batch, C{i}_ADDR_c, C_golden + {i} * 64);\n"
        code_string +=  "\t\t}\n"
        code_string +=  "\t\tsnrt_cluster_hw_barrier();\n"
        code_string +=  "\t\tif(snrt_is_dm_core()){\n"
        code_string += f"\t\t   snrt_dma_start_1d(ERROR + {i}, local_err_{i}, sizeof(uint32_t));\n"
        code_string +=  "\t\t}\n"
        code_string +=  "\t}"
        code_string += emit_global_barrier()
        
    return code_string

def emit_back_to_cva6():
    code_string = "\n"
    code_string += "\t//return to host\n"
    code_string += "\treturn_to_cva6(SYNC_ALL);\n"
    return code_string




def emit_header_file(**kwargs):
    addr_map = get_addr_map(**kwargs)
    result_map = get_result_map()
    emit_str = "#include \"data.h\"\n"
    emit_str += "#include \"snax-gemm-lib.h\"\n"
    emit_str += "#include \"snax-gemm-params.h\"\n"
    # emit_str += emit_check_result_function()
    emit_str += emit_global_ADDR(**kwargs)
    emit_str += emit_main_header()
    emit_str += emit_post_wakeup_cl()
    emit_str += emit_group_idx_comment(**kwargs)
    emit_str += emit_group_boundary(**kwargs)
    emit_str += emit_size_setting(**kwargs)
    emit_str += emit_subtraction_setting()
    emit_str += emit_init_TCDM(**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_broadcast_TCDM_addr(**kwargs)
    emit_str += emit_global_barrier()
    # emit_str += emit_init_GeMM_setup(**kwargs)
    # emit_str += emit_global_barrier()
    
    emit_str += emit_data_flow_table()

    emit_str += emit_start_gemm()

    # Start Loading the Weight from the L3
    # Assume the cluster all loading from L3
    # this is the setup phase hence will not logged by mcycle
    # emit_str += emit_init_loading_W(**kwargs)
    # emit_str += emit_global_barrier()
    # emit_str += emit_code_seperate_comment()
    
    # T0 
    # C0 Loading In0 from L3
    emit_str += emit_C0_load_from_L3("A",0,0,**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()    

    # T1
    # C0 Loading In1 from L3
    emit_str += emit_C0_load_from_L3("A",0,0,**kwargs)
  
    # C3 Loading I0 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()   
    # T2
    # C0 Loading In2 from L3:
    emit_str += emit_C0_load_from_L3("A",0,0,**kwargs)
    # C1 Loading Out0_0 from C0: Assume the GeMM out will go through the relu and 
    #                            be converted to int8
    #                            Right now i just use the GeMM Input A as the replacement of the GeMM output
    emit_str += emit_Cx_load_from_Cy(init_cluster=1,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    # C3 Loading In1 from Cluster0 : Again assume the Cluster has enough local memory to store the input 
    #                                So I just emphasize on the data transfer
    #                                And override the In each time
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 

    # T3
    # C0 loading In3 from L3
    emit_str += emit_C0_load_from_L3("A",0,0,**kwargs)
    # C1 loading Out0_1 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=1,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    # C3 Loading In2 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 

    # T4
    # C0 loading In4 from L3
    emit_str += emit_C0_load_from_L3("A",0,0,**kwargs)  
    # C1 loading Out0_2 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=1,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)                                                   
    # C2 loading Out1_0 from C1
    emit_str += emit_Cx_load_from_Cy(init_cluster=2,target_cluster=1,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs) 
    # C3 Loading In3 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T5       
    # C0 loading In5 from L3
    emit_str += emit_C0_load_from_L3("A",0,0,**kwargs)  
    # C1 loading Out0_3 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=1,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)   
    # C2 loading Out1_1 from C1
    emit_str += emit_Cx_load_from_Cy(init_cluster=2,target_cluster=1,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs) 
    # C3 Loading In4 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T6
    # C1 loading Out0_4 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=1,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)       
    # C2 loading Out1_2 from C1
    emit_str += emit_Cx_load_from_Cy(init_cluster=2,target_cluster=1,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
                        
    # C3 Loading In5 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    # C3 Loading out2_0 from C2
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=2,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)

    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T7
    # C1 loading Out0_5 from C0
    emit_str += emit_Cx_load_from_Cy(init_cluster=1,target_cluster=0,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs) 
    # C2 loading Out1_3 from C1
    emit_str += emit_Cx_load_from_Cy(init_cluster=2,target_cluster=1,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)                                         
    # C3 Loading out2_1 from C2
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=2,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T8
    # C2 loading Out1_4 from C1
    emit_str += emit_Cx_load_from_Cy(init_cluster=2,target_cluster=1,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)  
    # C3 Loading out2_2 from C2
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=2,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()


    # T9
    # C2 loading Out1_5 from C1
    emit_str += emit_Cx_load_from_Cy(init_cluster=2,target_cluster=1,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    # C3 Loading out2_3 from C2
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=2,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T10
    # C3 Loading out2_4 from C2
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=2,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)   
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T10
    # C3 Loading out2_5 from C2
    emit_str += emit_Cx_load_from_Cy(init_cluster=3,target_cluster=2,
                                    matrix="A",
                                    matrix_idx_x=0,matrix_idx_y=0,
                                    addr_map=addr_map,**kwargs)   

    emit_str += emit_end_gemm()

    # emit_str += emit_check_result(**kwargs)

    emit_str += emit_back_to_cva6()
    emit_str += emit_main_end()
    


    return emit_str








def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generate 2x2 GeMM for NoC")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    # Emit header file
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()