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
import math
import re
def deassemble_C00_work_string(work_string):
    pattern_matrix = r"\b[A-B]\d{2}\b"
    pattern_cluster = r"\bC\d{2}\b"
    match_matrix = re.findall(pattern_matrix, work_string)
    match_cluster = re.findall(pattern_cluster, work_string)

    matrix_string = match_matrix[0]
    cluster_string = match_cluster[0]

    matrix = matrix_string[0]
    matrix_idx_x = int(matrix_string[1])
    matrix_idx_y = int(matrix_string[2])

    cluster_num_x = int(cluster_string[1])
    cluster_num_y = int(cluster_string[2])
    return cluster_num_x, cluster_num_y, matrix,matrix_idx_x,matrix_idx_y
    
def deassemble_Cxy_work_string(work_string):
    pattern = r"(C\d{2}).*?([A-B]\d{2}).*?(C\d{2})"
    match = re.search(pattern, work_string)
    init_string = match.group(1)
    matrix_string = match.group(2)
    target_string = match.group(3)

    matrix = matrix_string[0]
    matrix_idx_x = int(matrix_string[1])
    matrix_idx_y = int(matrix_string[2])

    init_cluster_num_x = int(init_string[1])
    init_cluster_num_y = int(init_string[2])

    target_cluster_num_x = int(target_string[1])
    target_cluster_num_y = int(target_string[2])
    return init_cluster_num_x, init_cluster_num_y, target_cluster_num_x, target_cluster_num_y, matrix, matrix_idx_x, matrix_idx_y

def get_cluster_num(cluster_num_x,cluster_num_y,**kwargs):
    numX = kwargs["NUM_X"]
    numY = kwargs["NUM_Y"]    
    cluster_num = cluster_num_y + numX * cluster_num_x
    return cluster_num

def get_cluster_xy(cluster_num):
    cluster_x = cluster_num // 4
    cluster_y = cluster_num % 4
    return cluster_x, cluster_y

def emit_data_flow_table():
    code_string = "\n"
    code_string = """
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
    code_string = "\n"

    numX = kwargs["NUM_X"]
    numY = kwargs["NUM_Y"]
    for x in range(numX):
        for y in range(numY):
            code_string += f"uint8_t*  C{x}{y}_ADDR_a = 999;\n"
            code_string += f"uint8_t*  C{x}{y}_ADDR_b = 999;\n"
            # code_string += f"uint32_t* C{x}{y}_ADDR_c=0;\n"
    return code_string

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




def emit_init_Cxy_TCDM_ADDR_VARS(cluster_num_x,cluster_num_y,**kwargs):

    code_string = "\n"
    cluster_num = get_cluster_num(cluster_num_x,cluster_num_y,**kwargs)

    code_string += f"\t// TCDM Memory for Cluster{cluster_num_x}{cluster_num_y}\n"
    code_string += f"\tint32_t* local_err_C{cluster_num_x}{cluster_num_y};\n"

    code_string += f"\tint8_t*  local_a_C{cluster_num_x}{cluster_num_y};\n"
    code_string += f"\tint8_t*  local_b_C{cluster_num_x}{cluster_num_y};\n"
    code_string += f"\tint32_t* local_c_C{cluster_num_x}{cluster_num_y};\n"
    if((cluster_num_x==0) & (cluster_num_y==0)): 
        code_string += f"\tint8_t** local_addr_A_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint8_t** local_addr_B_C{cluster_num_x}{cluster_num_y};\n"
    elif ((cluster_num_x==0) & (cluster_num_y!=0)):
        code_string += f"\tint8_t** local_a_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint8_t** local_b_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint32_t** local_c_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y};\n"
    elif ((cluster_num_x!=0) & (cluster_num_y==0)):
        code_string += f"\tint8_t** local_a_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint8_t** local_b_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint32_t** local_c_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y};\n"
    else:
        code_string += f"\tint8_t** local_a_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint8_t** local_b_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint32_t** local_c_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint8_t** local_a_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint8_t** local_b_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y};\n"
        code_string += f"\tint32_t** local_c_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y};\n"        
     
    return code_string

def emit_init_TCDM_ADDR_VARS(**kwargs):
    code_string = "\n"
    numX = kwargs["NUM_X"]
    numY = kwargs["NUM_Y"]
    for x in range(numX):
        for y in range(numY):
            code_string +=emit_init_Cxy_TCDM_ADDR_VARS(x,y,**kwargs)
    return code_string






def emit_init_Cxy_TCDM_ADDR(cluster_num_x,cluster_num_y,**kwargs):
    delta_local_a = kwargs["delta_local_a"]
    delta_local_b = kwargs["delta_local_b"]
    cluster_num = get_cluster_num(cluster_num_x,cluster_num_y,**kwargs)
    code_string = "\n"
    code_string += f"\tif((snrt_global_core_idx()>=group{cluster_num}_bound_lower) && (snrt_global_core_idx()<group{cluster_num}_bound_upper))"
    code_string +=  "{\n"
    
    code_string +=  "\t\t// Start a new row to store err\n"
    code_string += f"\t\tlocal_err_C{cluster_num_x}{cluster_num_y} = (int32_t*)snrt_l1_next();\n"
    code_string += "\n"
    code_string +=  "\t\t// Space to store matrix\n"
    code_string += f"\t\tlocal_a_C{cluster_num_x}{cluster_num_y} = (int8_t*)(local_err_C{cluster_num_x}{cluster_num_y} + 256 * sizeof(int8_t));\n"
    code_string += f"\t\tlocal_b_C{cluster_num_x}{cluster_num_y} = local_a_C{cluster_num_x}{cluster_num_y} + {delta_local_a} * sizeof(int8_t);\n"
    code_string += f"\t\tlocal_c_C{cluster_num_x}{cluster_num_y} = (int32_t*)(local_b_C{cluster_num_x}{cluster_num_y} + {delta_local_b} * sizeof(int8_t));\n"
    code_string += "\n"
    if((cluster_num_x==0) & (cluster_num_y==0)): 
        code_string += f"\t\tlocal_addr_A_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_err_C{cluster_num_x}{cluster_num_y} + 1);\n"
        code_string += f"\t\tlocal_addr_B_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_addr_A_C{cluster_num_x}{cluster_num_y} + 1);\n"    
    elif((cluster_num_x==0) & (cluster_num_y!=0)):
        code_string += f"\t\tlocal_a_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_err_C{cluster_num_x}{cluster_num_y} + 1);\n"
        code_string += f"\t\tlocal_b_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_a_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y} + 1);\n"
    elif ((cluster_num_x!=0) & (cluster_num_y==0)):
        code_string += f"\t\tlocal_a_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_err_C{cluster_num_x}{cluster_num_y} + 1);\n"
        code_string += f"\t\tlocal_b_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_a_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y} + 1);\n"
    else:
        code_string += f"\t\tlocal_a_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_err_C{cluster_num_x}{cluster_num_y} + 1);\n"
        code_string += f"\t\tlocal_b_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_a_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y} + 1);\n"
        code_string += f"\t\tlocal_a_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_b_C{cluster_num_x}{cluster_num_y-1}_C{cluster_num_x}{cluster_num_y} + 1);\n"
        code_string += f"\t\tlocal_b_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y} = (int8_t**)(local_a_C{cluster_num_x-1}{cluster_num_y}_C{cluster_num_x}{cluster_num_y} + 1);\n"
    code_string += "\n"

    code_string += f"\t\t*local_err_C{cluster_num_x}{cluster_num_y} = 0;\n"
    if((cluster_num_x==0) & (cluster_num_y==0)): 
        code_string += f"\t\t*local_addr_A_C{cluster_num_x}{cluster_num_y} = A;\n"
        code_string += f"\t\t*local_addr_B_C{cluster_num_x}{cluster_num_y} = B;\n"
    code_string += f"\t\tC{cluster_num_x}{cluster_num_y}_ADDR_a = local_a_C{cluster_num_x}{cluster_num_y};\n"
    code_string += f"\t\tC{cluster_num_x}{cluster_num_y}_ADDR_b = local_b_C{cluster_num_x}{cluster_num_y};\n"     
    code_string += "\t}\n"

    return code_string

def emit_init_TCDM_ADDR(**kwargs):
    code_string = "\n"
    numX = kwargs["NUM_X"]
    numY = kwargs["NUM_Y"]
    for x in range(numX):
        for y in range(numY):
            code_string += emit_init_Cxy_TCDM_ADDR(x,y,**kwargs)
            # code_string += emit_global_barrier()
    return code_string


def emit_init_TCDM(**kwargs):
    code_string = "\n"
    code_string += "\t// Prepare addresses in TCDM\n"
    code_string += emit_init_TCDM_ADDR_VARS(**kwargs)
    code_string += "\t// Init Addresses in TCDM\n"
    code_string += emit_init_TCDM_ADDR(**kwargs)
    return code_string

def emit_broadcast_TCDM_addr(**kwargs):

    code_string = "\n"
    numX = kwargs["NUM_X"]
    numY = kwargs["NUM_Y"]
    code_string += "\t//Start the broadcasting of the tcdm address\n"
    for x in range(numX):
        for y in range(numY):
            if((x==0) & (y==0)):
                code_string += "\t//Cluster00 already got the matrix A and B addr\n"
                code_string += "\n"
            else:
                cluster_num = get_cluster_num(x,y,**kwargs)

                if((x==0) & (y!=0)):
                    code_string += f"\t//Cluster{x}{y} gets the TCDM address of the Cluster{x}{y-1}\n"
                    code_string += "\n"                    
                    code_string += f"\tif((snrt_global_core_idx()>=group{cluster_num}_bound_lower) && (snrt_global_core_idx()<group{cluster_num}_bound_upper))" 
                    code_string +=  "{\n"
                    code_string +=  "\t   if(snrt_is_dm_core()){\n"
                    # code_string += f"\t       *local_a_C{x}{y-1}_C{x}{y} = C{x}{y-1}_ADDR_a;\n"
                    # code_string += f"\t       *local_b_C{x}{y-1}_C{x}{y} = C{x}{y-1}_ADDR_b;\n"
                    code_string += f"\t     snrt_dma_start_1d(local_a_C{x}{y-1}_C{x}{y},&C{x}{y-1}_ADDR_a,4);\n"
                    code_string += f"\t     snrt_dma_start_1d(local_b_C{x}{y-1}_C{x}{y},&C{x}{y-1}_ADDR_b,4);\n"
                    code_string += f"\t     snrt_dma_wait_all();\n"
                    code_string +=  "\t     }\n"
                    code_string +=  "\t     snrt_cluster_hw_barrier();\n"

                    code_string +=  "\t}\n" 
                elif((x!=0) & (y==0)):
                    code_string += f"\t//Cluster{x}{y} gets the TCDM address of the Cluster{x-1}{y}\n"
                    code_string += "\n"                    
                    code_string += f"\tif((snrt_global_core_idx()>=group{cluster_num}_bound_lower) && (snrt_global_core_idx()<group{cluster_num}_bound_upper))" 
                    code_string +=  "{\n"
                    code_string +=  "\t   if(snrt_is_dm_core()){\n"
                    # code_string += f"\t       *local_a_C{x-1}{y}_C{x}{y} = C{x-1}{y}_ADDR_a;\n"
                    # code_string += f"\t       *local_b_C{x-1}{y}_C{x}{y} = C{x-1}{y}_ADDR_b;\n"
                    code_string += f"\t   snrt_dma_start_1d(local_a_C{x-1}{y}_C{x}{y},&C{x-1}{y}_ADDR_a,4);\n"
                    code_string += f"\t   snrt_dma_start_1d(local_b_C{x-1}{y}_C{x}{y},&C{x-1}{y}_ADDR_b,4);\n"
                    code_string += f"\t   snrt_dma_wait_all();\n"
                    code_string +=  "\t    }\n"            
                    code_string +=  "\t    snrt_cluster_hw_barrier();\n"
                    code_string +=  "\t}\n"
                else:
                    code_string += f"\t//Cluster{x}{y} gets the TCDM address of the Cluster{x-1}{y} and Cluster{x}{y-1}\n"
                    code_string += "\n"                    
                    code_string += f"\tif((snrt_global_core_idx()>=group{cluster_num}_bound_lower) && (snrt_global_core_idx()<group{cluster_num}_bound_upper))" 
                    code_string +=  "{\n"  
                    code_string +=  "\t  if(snrt_is_dm_core()){\n"
                    # code_string += f"\t       *local_a_C{x-1}{y}_C{x}{y} = C{x-1}{y}_ADDR_a;\n"
                    # code_string += f"\t       *local_b_C{x-1}{y}_C{x}{y} = C{x-1}{y}_ADDR_b;\n"
                    # code_string += f"\t       *local_a_C{x}{y-1}_C{x}{y} = C{x}{y-1}_ADDR_a;\n"
                    # code_string += f"\t       *local_b_C{x}{y-1}_C{x}{y} = C{x}{y-1}_ADDR_b;\n"
                    code_string += f"\t     snrt_dma_start_1d(local_a_C{x-1}{y}_C{x}{y},&C{x-1}{y}_ADDR_a,4);\n"
                    code_string += f"\t     snrt_dma_start_1d(local_b_C{x-1}{y}_C{x}{y},&C{x-1}{y}_ADDR_b,4);\n"
                    code_string += f"\t     snrt_dma_start_1d(local_a_C{x}{y-1}_C{x}{y},&C{x}{y-1}_ADDR_a,4);\n"
                    code_string += f"\t     snrt_dma_start_1d(local_b_C{x}{y-1}_C{x}{y},&C{x}{y-1}_ADDR_b,4);\n"                    
                    code_string += f"\t     snrt_dma_wait_all();\n"
                    code_string +=  "\t   }\n"
                    code_string +=  "\t   snrt_cluster_hw_barrier();\n"
                    code_string +=  "\t}\n"
                # code_string += emit_global_barrier()                   
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


def emit_Cx_load_A_from_L3(cluster_num_x,cluster_num_y,A_idx_x,A_idx_y,**kwargs):
    Batch = kwargs["Batch"]
    strideA = kwargs["strideA"]
    strideInnermostA = kwargs["strideInnermostA"]
    meshRow = kwargs["meshRow"]
    tileSize = kwargs["tileSize"]
    gemm_idx_m, gemm_idx_k = A_idx_2_gemm_idx(A_idx_x,A_idx_y)

    delta_addr_a = f"{A_idx_y} * {strideInnermostA}"
    delta_addr_A = get_delta_addr_A(gemm_idx_m,gemm_idx_k,**kwargs)
    stride_dst = strideA 
    stride_src = kwargs["M"] * kwargs["meshRow"] * kwargs["tileSize"] * kwargs["K"]
    matrix_str = f'A{A_idx_x}{A_idx_y}' 
    cluster_num = get_cluster_num(cluster_num_x,cluster_num_y,**kwargs)
    code_string = "\n"
    code_string += f"\t// Cluster {cluster_num_x}{cluster_num_y} Load the {matrix_str} from L3\n"
    code_string += f"\tif((snrt_global_core_idx()>=group{cluster_num}_bound_lower) && (snrt_global_core_idx()<group{cluster_num}_bound_upper))"
    code_string +=  "{\n"
    code_string +=  "\t  if (snrt_is_dm_core()) {\n"
    code_string += f"\t     uint32_t dma_C{cluster_num_x}{cluster_num_y}_load_{matrix_str}_start = mcycle();\n"
    code_string += f"\t    snrt_dma_start_2d(local_a_C{cluster_num_x}{cluster_num_y} + {delta_addr_a}, *local_addr_A_C{cluster_num_x}{cluster_num_y} + {delta_addr_A}, {meshRow} * {tileSize} * sizeof(int8_t), {stride_dst}, {stride_src}, {Batch});\n"
    code_string += f"\t    snrt_dma_wait_all();\n"
    code_string += f"\t    uint32_t dma_C{cluster_num_x}{cluster_num_y}_load_{matrix_str}_end = mcycle();\n"
    code_string +=  "\t    }\n"
    code_string +=  "\t}\n"
    code_string +=  "\n"
    return code_string


def emit_Cx_load_B_from_L3(cluster_num_x,cluster_num_y,B_idx_x,B_idx_y,**kwargs):
    Batch = kwargs["Batch"]
    strideB = kwargs["strideB"]
    strideInnermostB = kwargs["strideInnermostB"]
    meshCol = kwargs["meshCol"]
    tileSize = kwargs["tileSize"]
    gemm_idx_n, gemm_idx_k = B_idx_2_gemm_idx(B_idx_x,B_idx_y)

    delta_addr_b = f"{B_idx_x} * {strideInnermostB}"
    delta_addr_B = get_delta_addr_B(gemm_idx_n,gemm_idx_k,**kwargs)
    stride_dst = strideB
    stride_src = kwargs["K"] * kwargs["tileSize"] * kwargs["meshCol"] * kwargs["N"]
    matrix_str = f'B{B_idx_x}{B_idx_y}' 
    cluster_num = get_cluster_num(cluster_num_x,cluster_num_y,**kwargs)
    code_string = "\n"
    code_string += f"\t// Cluster {cluster_num_x}{cluster_num_y} Load the {matrix_str} from L3\n"
    code_string += f"\tif((snrt_global_core_idx()>=group{cluster_num}_bound_lower) && (snrt_global_core_idx()<group{cluster_num}_bound_upper))"
    code_string +=  "{\n"
    code_string +=  "\t  if (snrt_is_dm_core()) {\n"
    code_string += f"\t     uint32_t dma_C{cluster_num_x}{cluster_num_y}_load_{matrix_str}_start = mcycle();\n"
    code_string += f"\t    snrt_dma_start_2d(local_b_C{cluster_num_x}{cluster_num_y} + {delta_addr_b}, *local_addr_B_C{cluster_num_x}{cluster_num_y} + {delta_addr_B}, {meshCol} * {tileSize} * sizeof(int8_t), {stride_dst}, {stride_src}, {Batch});\n"
    code_string += f"\t    snrt_dma_wait_all();\n"
    code_string += f"\t    uint32_t dma_C{cluster_num_x}{cluster_num_y}_load_{matrix_str}_end = mcycle();\n"
    code_string +=  "\t    }\n"
    code_string +=  "\t}\n"
    code_string +=  "\n"
    return code_string



def emit_Cxy_load_from_L3(work_string,**kwargs):
    code_string = "\n"
    cluster_num_x,cluster_num_y,matrix,matrix_idx_x,matrix_idx_y = deassemble_C00_work_string(work_string)
    if(matrix=="A"):
        code_string += emit_Cx_load_A_from_L3(cluster_num_x,cluster_num_y,matrix_idx_x,matrix_idx_y,**kwargs)
    if(matrix=="B"):
        code_string += emit_Cx_load_B_from_L3(cluster_num_x,cluster_num_y,matrix_idx_x,matrix_idx_y,**kwargs)
    return code_string


def emit_Cxy_load_A_from_Cxy(init_cluster_num_x,init_cluster_num_y,
                             target_cluster_num_x,target_cluster_num_y,
                             A_idx_x,A_idx_y,
                             **kwargs):
    init_cluster_num = get_cluster_num(init_cluster_num_x,init_cluster_num_y,**kwargs)
    Batch = kwargs["Batch"]
    meshRow = kwargs["meshRow"]
    tileSize = kwargs["tileSize"]
    strideA = kwargs["strideA"]
    strideInnermostA = kwargs["strideInnermostA"]
    matrix_str = f'A{A_idx_x}{A_idx_y}'
    init_cluster_str = f'C{init_cluster_num_x}{init_cluster_num_y}'
    target_cluster_str = f'C{target_cluster_num_x}{target_cluster_num_y}'
    
    dma_src_addr = f"*local_a_C{target_cluster_num_x}{target_cluster_num_y}_C{init_cluster_num_x}{init_cluster_num_y} + {A_idx_y} * {strideInnermostA}"
    dma_dst_addr = f"local_a_C{init_cluster_num_x}{init_cluster_num_y} + {A_idx_y} * {strideInnermostA}"

    
    code_string = "\n"
    code_string += f"\t// {init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}\n"
    code_string += f"\tif((snrt_global_core_idx()>=group{init_cluster_num}_bound_lower) && (snrt_global_core_idx()<group{init_cluster_num}_bound_upper))"
    code_string +=  "{\n"
    code_string +=  "\t\tif(snrt_is_dm_core()){\n"
    code_string += f"\t\t\tuint32_t dma_{init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}_start = mcycle();\n"
    code_string += f"\t\t\tsnrt_dma_start_2d({dma_dst_addr}, {dma_src_addr}, {meshRow} * {tileSize} * sizeof(int8_t), {strideA}, {strideA}, {Batch});\n"
    code_string +=  "\t\t\tsnrt_dma_wait_all();\n"
    code_string += f"\t\t\tuint32_t dma_{init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}_end = mcycle();\n"
    code_string +=  "\t\t}\n"
    code_string +=  "\t}\n"
    return code_string

def emit_Cxy_load_B_from_Cxy(init_cluster_num_x,init_cluster_num_y,
                             target_cluster_num_x,target_cluster_num_y,
                             B_idx_x,B_idx_y,
                             **kwargs):
    init_cluster_num = get_cluster_num(init_cluster_num_x,init_cluster_num_y,**kwargs)
    strideInnermostB = kwargs["strideInnermostB"]
    Batch = kwargs["Batch"]
    meshCol = kwargs["meshCol"]
    tileSize = kwargs["tileSize"]
    strideB = kwargs["strideB"]

    matrix_str = f'B{B_idx_x}{B_idx_y}'
    init_cluster_str = f'C{init_cluster_num_x}{init_cluster_num_y}'
    target_cluster_str = f'C{target_cluster_num_x}{target_cluster_num_y}'
    
    dma_src_addr = f"*local_b_C{target_cluster_num_x}{target_cluster_num_y}_C{init_cluster_num_x}{init_cluster_num_y} + {B_idx_x} * {strideInnermostB}"
    dma_dst_addr = f"local_b_C{init_cluster_num_x}{init_cluster_num_y} + {B_idx_x} * {strideInnermostB}"
    
    code_string = "\n"
    code_string += f"\t// {init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}\n"
    code_string += f"\tif((snrt_global_core_idx()>=group{init_cluster_num}_bound_lower) && (snrt_global_core_idx()<group{init_cluster_num}_bound_upper))"
    code_string +=  "{\n"
    code_string +=  "\t\tif(snrt_is_dm_core()){\n"
    code_string += f"\t\t\tuint32_t dma_{init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}_start = mcycle();\n"
    code_string += f"\t\t\tsnrt_dma_start_2d({dma_dst_addr}, {dma_src_addr}, {meshCol} * {tileSize} * sizeof(int8_t), {strideB}, {strideB}, {Batch});\n"
    code_string +=  "\t\t\tsnrt_dma_wait_all();\n"
    code_string += f"\t\t\tuint32_t dma_{init_cluster_str}_load_{matrix_str}_from_{target_cluster_str}_end = mcycle();\n"
    code_string +=  "\t\t}\n"
    code_string +=  "\t}\n"
    return code_string

def emit_Cxy_load_from_Cxy(work_string,**kwargs):
    init_cluster_num_x, init_cluster_num_y,target_cluster_num_x, target_cluster_num_y,matrix,matrix_idx_x,matrix_idx_y = deassemble_Cxy_work_string(work_string)
    code_string = "\n"
    if(matrix=="A"):
        code_string += emit_Cxy_load_A_from_Cxy(init_cluster_num_x, init_cluster_num_y,
                                                target_cluster_num_x, target_cluster_num_y,
                                                matrix_idx_x,matrix_idx_y,
                                                **kwargs)
    if(matrix=="B"):
        code_string += emit_Cxy_load_B_from_Cxy(init_cluster_num_x, init_cluster_num_y,
                                                target_cluster_num_x, target_cluster_num_y,
                                                matrix_idx_x,matrix_idx_y,
                                                **kwargs)
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
    # addr_map = get_addr_map(**kwargs)
    # result_map = get_result_map()
    emit_str = "#include \"data.h\"\n"
    emit_str += "#include \"snax-gemm-lib.h\"\n"
    emit_str += "#include \"snax-gemm-params.h\"\n"
    # emit_str += emit_check_result_function()
    emit_str += emit_global_ADDR(**kwargs)
    emit_str += emit_main_header()
    emit_str += emit_post_wakeup_cl()
    emit_str += emit_group_idx_comment(**kwargs)
    emit_str += emit_group_boundary(**kwargs)
    # emit_str += emit_size_setting(**kwargs)
    # emit_str += emit_subtraction_setting()
    emit_str += emit_init_TCDM(**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_broadcast_TCDM_addr(**kwargs)
    # emit_str += emit_init_GeMM_setup(**kwargs)
    emit_str += emit_global_barrier()
    
    emit_str += emit_data_flow_table()

    emit_str += emit_start_gemm()

    # T0
    # C00 Load A00 from L3
    emit_str += "\t//T0:\n"
    emit_str += emit_Cxy_load_from_L3("C00 Load A00 from L3", **kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T1
    emit_str += "\t//T1:\n"
    # C00 Load A01 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A01 from L3", **kwargs)
    # C01 Load A00 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load A00 from C00",**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T2
    emit_str += "\t//T2:\n"
    # C00 Load A02 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A02 from L3", **kwargs)
    # C01 Load A01 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load A01 from C00",**kwargs)
    # C02 Load A00 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load A00 from C01",**kwargs)    
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T3
    emit_str += "\t//T3:\n"
    # C00 Load A03 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A03 from L3", **kwargs)
    # C01 Load A02 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load A02 from C00",**kwargs)
    # C02 Load A01 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load A01 from C01",**kwargs) 
    # C03 Load A00 from C01
    emit_str += emit_Cxy_load_from_Cxy("C03 Load A00 from C02",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T4
    emit_str += "\t//T4:\n"
    # C00 Load B00 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B00 from L3", **kwargs)
    # C01 Load A03 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load A03 from C00",**kwargs)
    # C02 Load A02 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load A02 from C01",**kwargs) 
    # C03 Load A01 from C01
    emit_str += emit_Cxy_load_from_Cxy("C03 Load A01 from C02",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T5
    emit_str += "\t//T5:\n"
    # C00 Load B10 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B10 from L3", **kwargs)
    # C02 Load A03 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load A03 from C01",**kwargs) 
    # C03 Load A02 from C01
    emit_str += emit_Cxy_load_from_Cxy("C03 Load A02 from C02",**kwargs) 
    # C10 Load B00 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load B00 from C00",**kwargs)         
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T6
    emit_str += "\t//T6:\n"
    # C00 Load B20 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B20 from L3", **kwargs)
    # C03 Load A03 from C01
    emit_str += emit_Cxy_load_from_Cxy("C03 Load A03 from C02",**kwargs) 
    # C10 Load B10 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load B10 from C00",**kwargs)
    # C20 Load B00 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load B00 from C10",**kwargs)  
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()    
                                   
    # T7
    emit_str += "\t//T7:\n"
    # C00 Load B30 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B30 from L3", **kwargs)
    # C10 Load B20 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load B20 from C00",**kwargs) 
    # C20 Load B10 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load B10 from C10",**kwargs)
    # C30 Load B00 from C20
    emit_str += emit_Cxy_load_from_Cxy("C30 Load B00 from C20",**kwargs)    
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()  

    # T8
    emit_str += "\t//T8:\n"
    # C00 Load A10 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A10 from L3", **kwargs)
    # C10 Load B30 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load B30 from C00",**kwargs) 
    # C20 Load B20 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load B20 from C10",**kwargs)
    # C30 Load B10 from C20
    emit_str += emit_Cxy_load_from_Cxy("C30 Load B10 from C20",**kwargs)    
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 

    # T9
    emit_str += "\t//T9:\n"
    # C00 Load A11 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A11 from L3", **kwargs)
    # C10 Load A10 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A10 from C00",**kwargs) 
    # C20 Load B30 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load B30 from C10",**kwargs)
    # C30 Load B20 from C20
    emit_str += emit_Cxy_load_from_Cxy("C30 Load B20 from C20",**kwargs)    
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 

    # T10
    emit_str += "\t//T10:\n"
    # C00 Load A12 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A12 from L3", **kwargs)
    # C10 Load A11 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A11 from C00",**kwargs)
    # C11 Load A10 from C10
    emit_str += emit_Cxy_load_from_Cxy("C11 Load A10 from C10",**kwargs)
    # C30 Load B30 from C20
    emit_str += emit_Cxy_load_from_Cxy("C30 Load B30 from C20",**kwargs) 

    # T11
    emit_str += "\t//T11:\n"
    # C00 Load A13 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A13 from L3", **kwargs)
    # C10 Load A12 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A12 from C00",**kwargs)
    # C11 Load A11 from C10
    emit_str += emit_Cxy_load_from_Cxy("C11 Load A11 from C10",**kwargs)
    # C12 Load A10 from C11
    emit_str += emit_Cxy_load_from_Cxy("C12 Load A10 from C11",**kwargs)
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 

    # T12
    emit_str += "\t//T12:\n"
    # C00 Load B01 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B01 from L3", **kwargs)
    # C10 Load A13 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A13 from C00",**kwargs)
    # C11 Load A12 from C10
    emit_str += emit_Cxy_load_from_Cxy("C11 Load A12 from C10",**kwargs)
    # C12 Load A11 from C11
    emit_str += emit_Cxy_load_from_Cxy("C12 Load A11 from C11",**kwargs)
    # C13 Load A10 from C12
    emit_str += emit_Cxy_load_from_Cxy("C13 Load A10 from C12",**kwargs)    
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 

    # T13
    emit_str += "\t//T13:\n"
    # C00 Load B11 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B11 from L3", **kwargs)
    # C01 Load B01 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B01 from C00", **kwargs)
    # C11 Load A13 from C10
    emit_str += emit_Cxy_load_from_Cxy("C11 Load A13 from C10",**kwargs)
    # C12 Load A12 from C11
    emit_str += emit_Cxy_load_from_Cxy("C12 Load A12 from C11",**kwargs)
    # C13 Load A11 from C12
    emit_str += emit_Cxy_load_from_Cxy("C13 Load A11 from C12",**kwargs)    
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 


    # T14
    emit_str += "\t//T14:\n"
    # C00 Load B21 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B21 from L3", **kwargs)
    # C01 Load B11 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B11 from C00", **kwargs)
    # C11 Load B01 from C01
    emit_str += emit_Cxy_load_from_Cxy("C11 Load B01 from C01",**kwargs)
    # C12 Load A13 from C11
    emit_str += emit_Cxy_load_from_Cxy("C12 Load A13 from C11",**kwargs)
    # C13 Load A12 from C12
    emit_str += emit_Cxy_load_from_Cxy("C13 Load A12 from C12",**kwargs)    
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 

    # T15
    emit_str += "\t//T15:\n"
    # C00 Load B31 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B31 from L3", **kwargs)
    # C01 Load B21 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B21 from C00", **kwargs)
    # C11 Load B11 from C01
    emit_str += emit_Cxy_load_from_Cxy("C11 Load B11 from C01",**kwargs)
    # C13 Load A13 from C12
    emit_str += emit_Cxy_load_from_Cxy("C13 Load A13 from C12",**kwargs) 
    # C21 Load B01 from C11
    emit_str += emit_Cxy_load_from_Cxy("C21 Load B01 from C11",**kwargs)    

    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T16
    emit_str += "\t//T16:\n"
    # C00 Load A20 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A20 from L3", **kwargs)
    # C01 Load B31 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B31 from C00", **kwargs)
    # C11 Load B21 from C01
    emit_str += emit_Cxy_load_from_Cxy("C11 Load B21 from C01",**kwargs)
    # C21 Load B11 from C11
    emit_str += emit_Cxy_load_from_Cxy("C21 Load B11 from C11",**kwargs) 
    # C31 Load B01 from C21
    emit_str += emit_Cxy_load_from_Cxy("C31 Load B01 from C21",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 

    # T17
    emit_str += "\t//T17:\n"
    # C00 Load A21 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A21 from L3", **kwargs)
    # C10 Load A20 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A20 from C00", **kwargs)
    # C11 Load B31 from C01
    emit_str += emit_Cxy_load_from_Cxy("C11 Load B31 from C01",**kwargs)
    # C21 Load B21 from C11
    emit_str += emit_Cxy_load_from_Cxy("C21 Load B21 from C11",**kwargs) 
    # C31 Load B11 from C21
    emit_str += emit_Cxy_load_from_Cxy("C31 Load B11 from C21",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T18
    emit_str += "\t//T18:\n"
    # C00 Load A22 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A22 from L3", **kwargs)
    # C10 Load A21 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A21 from C00", **kwargs)
    # C20 Load A20 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load A20 from C10", **kwargs)
    # C21 Load B31 from C11
    emit_str += emit_Cxy_load_from_Cxy("C21 Load B31 from C11",**kwargs) 
    # C31 Load B21 from C21
    emit_str += emit_Cxy_load_from_Cxy("C31 Load B21 from C21",**kwargs) 

    # T19
    emit_str += "\t//T19:\n"
    # C00 Load A23 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A23 from L3", **kwargs)
    # C10 Load A22 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A22 from C00", **kwargs)
    # C20 Load A21 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load A21 from C10", **kwargs)
    # C21 Load A20 from C20
    emit_str += emit_Cxy_load_from_Cxy("C21 Load A20 from C20",**kwargs) 
    # C31 Load B31 from C21
    emit_str += emit_Cxy_load_from_Cxy("C31 Load B31 from C21",**kwargs) 
    
    # T20
    emit_str += "\t//T20:\n"
    # C00 Load B02 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B02 from L3", **kwargs)
    # C10 Load A23 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A23 from C00", **kwargs)
    # C20 Load A22 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load A22 from C10", **kwargs)
    # C21 Load A21 from C20
    emit_str += emit_Cxy_load_from_Cxy("C21 Load A21 from C20",**kwargs) 
    # C22 Load A20 from C21
    emit_str += emit_Cxy_load_from_Cxy("C22 Load A20 from C21",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()             

    # T21
    emit_str += "\t//T21:\n"
    # C00 Load B12 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B12 from L3", **kwargs)
    # C01 Load B02 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B02 from C00", **kwargs)
    # C20 Load A23 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load A23 from C10", **kwargs)
    # C21 Load A22 from C20
    emit_str += emit_Cxy_load_from_Cxy("C21 Load A22 from C20",**kwargs) 
    # C22 Load A21 from C21
    emit_str += emit_Cxy_load_from_Cxy("C22 Load A21 from C21",**kwargs) 
    # C23 Load A20 from C22
    emit_str += emit_Cxy_load_from_Cxy("C23 Load A20 from C22",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T22
    emit_str += "\t//T22:\n"
    # C00 Load B22 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B22 from L3", **kwargs)
    # C01 Load B12 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B12 from C00", **kwargs)
    # C02 Load B02 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load B02 from C01", **kwargs)
    # C21 Load A23 from C20
    emit_str += emit_Cxy_load_from_Cxy("C21 Load A23 from C20",**kwargs) 
    # C22 Load A22 from C21
    emit_str += emit_Cxy_load_from_Cxy("C22 Load A22 from C21",**kwargs) 
    # C23 Load A21 from C22
    emit_str += emit_Cxy_load_from_Cxy("C23 Load A21 from C22",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()   

    # T23
    emit_str += "\t//T23:\n"
    # C00 Load B32 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B32 from L3", **kwargs)
    # C01 Load B22 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B22 from C00", **kwargs)
    # C02 Load B12 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load B12 from C01", **kwargs)
    # C12 Load B02 from C02
    emit_str += emit_Cxy_load_from_Cxy("C12 Load B02 from C02", **kwargs)
    # C22 Load A23 from C21
    emit_str += emit_Cxy_load_from_Cxy("C22 Load A23 from C21",**kwargs) 
    # C23 Load A22 from C22
    emit_str += emit_Cxy_load_from_Cxy("C23 Load A22 from C22",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment() 

    # T24
    emit_str += "\t//T24:\n"
    # C00 Load A30 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A30 from L3", **kwargs)
    # C01 Load B32 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B32 from C00", **kwargs)
    # C02 Load B22 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load B22 from C01", **kwargs)
    # C12 Load B12 from C02
    emit_str += emit_Cxy_load_from_Cxy("C12 Load B12 from C02", **kwargs)
    # C22 Load B02 from C12
    emit_str += emit_Cxy_load_from_Cxy("C22 Load B02 from C12",**kwargs) 
    # C23 Load A23 from C22
    emit_str += emit_Cxy_load_from_Cxy("C23 Load A23 from C22",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T25
    emit_str += "\t//T25:\n"
    # C00 Load A31 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A31 from L3", **kwargs)
    # C02 Load B32 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load B32 from C01", **kwargs)
    # C10 Load A30 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A30 from C00", **kwargs)
    # C12 Load B22 from C02
    emit_str += emit_Cxy_load_from_Cxy("C12 Load B22 from C02", **kwargs)
    # C22 Load B12 from C12
    emit_str += emit_Cxy_load_from_Cxy("C22 Load B12 from C12",**kwargs) 
    # C32 Load B02 from C22
    emit_str += emit_Cxy_load_from_Cxy("C32 Load B02 from C22",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T26
    emit_str += "\t//T26:\n"
    # C00 Load A32 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A32 from L3", **kwargs)
    # C10 Load A31 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A31 from C00", **kwargs)
    # C12 Load B32 from C02
    emit_str += emit_Cxy_load_from_Cxy("C12 Load B32 from C02", **kwargs)
    # C20 Load A30 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load A30 from C10", **kwargs)
    # C22 Load B22 from C12
    emit_str += emit_Cxy_load_from_Cxy("C22 Load B22 from C12",**kwargs) 
    # C32 Load B12 from C22
    emit_str += emit_Cxy_load_from_Cxy("C32 Load B12 from C22",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T27
    emit_str += "\t//T27:\n"
    # C00 Load A33 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load A33 from L3", **kwargs)
    # C10 Load A32 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A32 from C00", **kwargs)
    # C20 Load A31 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load A31 from C10", **kwargs)
    # C22 Load B32 from C12
    emit_str += emit_Cxy_load_from_Cxy("C22 Load B32 from C12",**kwargs) 
    # C30 Load A30 from C20
    emit_str += emit_Cxy_load_from_Cxy("C30 Load A30 from C20",**kwargs) 
    # C32 Load B22 from C22
    emit_str += emit_Cxy_load_from_Cxy("C32 Load B22 from C22",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T28
    emit_str += "\t//T28:\n"
    # C00 Load B03 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B03 from L3", **kwargs)
    # C10 Load A33 from C00
    emit_str += emit_Cxy_load_from_Cxy("C10 Load A33 from C00", **kwargs)
    # C20 Load A32 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load A32 from C10", **kwargs)
    # C30 Load A31 from C20
    emit_str += emit_Cxy_load_from_Cxy("C30 Load A31 from C20",**kwargs)
    # C31 Load A30 from C30
    emit_str += emit_Cxy_load_from_Cxy("C31 Load A30 from C30",**kwargs) 
    # C32 Load B32 from C22
    emit_str += emit_Cxy_load_from_Cxy("C32 Load B32 from C22",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T29
    emit_str += "\t//T29:\n"
    # C00 Load B13 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B13 from L3", **kwargs)
    # C01 Load B03 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B03 from C00", **kwargs)
    # C20 Load A33 from C10
    emit_str += emit_Cxy_load_from_Cxy("C20 Load A33 from C10", **kwargs)
    # C30 Load A32 from C20
    emit_str += emit_Cxy_load_from_Cxy("C30 Load A32 from C20",**kwargs)
    # C31 Load A31 from C30
    emit_str += emit_Cxy_load_from_Cxy("C31 Load A31 from C30",**kwargs) 
    # C32 Load A30 from C31
    emit_str += emit_Cxy_load_from_Cxy("C32 Load A30 from C31",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T30
    emit_str += "\t//T30:\n"
    # C00 Load B23 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B23 from L3", **kwargs)
    # C01 Load B13 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B13 from C00", **kwargs)
    # C02 Load B03 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load B03 from C01", **kwargs)

    # C30 Load A33 from C20
    emit_str += emit_Cxy_load_from_Cxy("C30 Load A33 from C20", **kwargs)
    # C31 Load A32 from C20
    emit_str += emit_Cxy_load_from_Cxy("C31 Load A32 from C30",**kwargs)
    # C32 Load A31 from C31
    emit_str += emit_Cxy_load_from_Cxy("C32 Load A31 from C31",**kwargs) 
    # C33 Load A30 from C32
    emit_str += emit_Cxy_load_from_Cxy("C33 Load A30 from C32",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T31
    emit_str += "\t//T31:\n"
    # C00 Load B33 from L3
    emit_str += emit_Cxy_load_from_L3("C00 Load B33 from L3", **kwargs)
    # C01 Load B23 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B23 from C00", **kwargs)
    # C02 Load B13 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load B13 from C01", **kwargs)
    # C03 Load B03 from C02
    emit_str += emit_Cxy_load_from_Cxy("C03 Load B03 from C02", **kwargs)    
    # C31 Load A33 from C30
    emit_str += emit_Cxy_load_from_Cxy("C31 Load A33 from C30",**kwargs)
    # C32 Load A32 from C31
    emit_str += emit_Cxy_load_from_Cxy("C32 Load A32 from C31",**kwargs) 
    # C33 Load A31 from C32
    emit_str += emit_Cxy_load_from_Cxy("C33 Load A31 from C32",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T32
    emit_str += "\t//T32:\n"
    # C01 Load B33 from C00
    emit_str += emit_Cxy_load_from_Cxy("C01 Load B33 from C00", **kwargs)
    # C02 Load B23 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load B23 from C01", **kwargs)
    # C03 Load B13 from C02
    emit_str += emit_Cxy_load_from_Cxy("C03 Load B13 from C02", **kwargs)   
    # C13 Load B03 from C03
    emit_str += emit_Cxy_load_from_Cxy("C13 Load B03 from C03", **kwargs)  
    # C32 Load A33 from C31
    emit_str += emit_Cxy_load_from_Cxy("C32 Load A33 from C31",**kwargs) 
    # C33 Load A32 from C32
    emit_str += emit_Cxy_load_from_Cxy("C33 Load A32 from C32",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()
    
    # T33
    emit_str += "\t//T33:\n"
    # C02 Load B23 from C01
    emit_str += emit_Cxy_load_from_Cxy("C02 Load B33 from C01", **kwargs)
    # C03 Load B23 from C02
    emit_str += emit_Cxy_load_from_Cxy("C03 Load B23 from C02", **kwargs)   
    # C13 Load B13 from C03
    emit_str += emit_Cxy_load_from_Cxy("C13 Load B13 from C03", **kwargs)  
    # C23 Load B03 from C13
    emit_str += emit_Cxy_load_from_Cxy("C23 Load B03 from C13", **kwargs) 
    # C33 Load A33 from C32
    emit_str += emit_Cxy_load_from_Cxy("C33 Load A33 from C32",**kwargs) 
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T34
    emit_str += "\t//T34:\n"
    # C03 Load B33 from C02
    emit_str += emit_Cxy_load_from_Cxy("C03 Load B33 from C02", **kwargs)   
    # C13 Load B23 from C03
    emit_str += emit_Cxy_load_from_Cxy("C13 Load B23 from C03", **kwargs)  
    # C23 Load B13 from C13
    emit_str += emit_Cxy_load_from_Cxy("C23 Load B13 from C13", **kwargs) 
    # C33 Load B03 from C23
    emit_str += emit_Cxy_load_from_Cxy("C33 Load B03 from C23", **kwargs)     

    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T35
    emit_str += "\t//T35:\n"
    # C13 Load B33 from C03
    emit_str += emit_Cxy_load_from_Cxy("C13 Load B33 from C03", **kwargs)  
    # C23 Load B23 from C13
    emit_str += emit_Cxy_load_from_Cxy("C23 Load B23 from C13", **kwargs) 
    # C33 Load B13 from C23
    emit_str += emit_Cxy_load_from_Cxy("C33 Load B13 from C23", **kwargs)     

    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T36
    emit_str += "\t//T36:\n"
    # C23 Load B33 from C13
    emit_str += emit_Cxy_load_from_Cxy("C23 Load B33 from C13", **kwargs) 
    # C33 Load B23 from C23
    emit_str += emit_Cxy_load_from_Cxy("C33 Load B23 from C23", **kwargs)     
    emit_str += emit_global_barrier()
    emit_str += emit_code_seperate_comment()

    # T37
    emit_str += "\t//T37:\n"
    # C33 Load B33 from C23
    emit_str += emit_Cxy_load_from_Cxy("C33 Load B33 from C23", **kwargs)    


    emit_str += emit_end_gemm()
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