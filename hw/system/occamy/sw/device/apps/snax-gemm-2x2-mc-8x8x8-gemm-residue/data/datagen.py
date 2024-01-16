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
# # Add data utility path
# sys.path.append(
#     os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/")
# )
# from data_utils import (format_scalar_definition, format_vector_definition)  # noqa E402

np.random.seed(42)

def variable_attributes(alignment=None, section=None):
    attributes = ''
    if alignment:
        attributes = f'__attribute__ ((aligned ({alignment})))'
    if section:
        attributes += f' __attribute__ ((section ("{section}")))'
    return attributes

def format_scalar_definition(type, uid, scalar):
    s = f'{type} {uid} = {scalar};'
    return s
def format_vector_definition(type, uid, vector, alignment=None, section=None):
    attributes = variable_attributes(alignment, section)
    s = f'{type} {uid}[{len(vector)}] {attributes} = ' + '{\n'
    for el in vector:
        if type != 'char':
            el_str = f'{el}'
        else:
            el_str = f'0x{el:02x}'
        s += f'\t{el_str},\n'
    s += '};'
    return s

# Golden model in python
def block_gemm_golden_model(Batch, m, k, n, row, size, col, a, b,
                            subtraction_a, subtraction_b):
    c = np.zeros(Batch * m * row * n * col, dtype=(np.int32))
    for bb in range(Batch):
        for mm in range(m):
            for nn in range(n):
                for kk in range(k):
                    for rr in range(row):
                        for cc in range(col):
                            for ss in range(size):
                                c_index = (
                                    bb * m * n * row * col
                                    + mm * n * row * col
                                    + nn * row * col
                                    + rr * col
                                    + cc
                                )
                                a_index = (
                                    bb * m * k * row * size
                                    + mm * k * row * size
                                    + kk * row * size
                                    + rr * size
                                    + ss
                                )
                                b_index = (
                                    bb * n * k * size * col
                                    + nn * k * size * col
                                    + kk * size * col
                                    + cc * size
                                    + ss
                                )
                                c[c_index] = c[c_index] + \
                                    (a[a_index] - subtraction_a) * \
                                    (b[b_index] - subtraction_b)
    return c


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    # emit_str += "#include \"snax-gemm-layer.h\"\n\n"
    emit_str += emit_gemm_data(**kwargs)
    emit_str += "\n"
    emit_str += "uint32_t ERROR[4] = {0,0,0,0};"
    # emit_str += emit_gemm_struct()
    return emit_str

def emit_gemm_struct():
    struct_str = "\n\n"
    struct_str += "snax_gemm_layer snax_gemm_layer_l;\n\n"
    struct_str += "snax_gemm_layer_l.Batch = Batch; \n"
    struct_str += "snax_gemm_layer_l.M = M; \n"
    struct_str += "snax_gemm_layer_l.K = K; \n"
    struct_str += "snax_gemm_layer_l.N = N; \n"
    struct_str += "snax_gemm_layer_l.strideInnermostA = strideInnermostA; \n"
    struct_str += "snax_gemm_layer_l.strideInnermostB = strideInnermostB; \n"
    struct_str += "snax_gemm_layer_l.strideInnermostC = strideInnermostC; \n"
    struct_str += "snax_gemm_layer_l.ldA = ldA; \n"
    struct_str += "snax_gemm_layer_l.ldB = ldB; \n"
    struct_str += "snax_gemm_layer_l.ldC = ldC; \n"
    struct_str += "snax_gemm_layer_l.strideA = strideA; \n"
    struct_str += "snax_gemm_layer_l.strideB = strideB; \n"
    struct_str += "snax_gemm_layer_l.strideC = strideC; \n"
    struct_str += "snax_gemm_layer_l.subtraction_a = subtraction_a; \n"
    struct_str += "snax_gemm_layer_l.subtraction_b = subtraction_b; \n"
    struct_str += "\n"
    return struct_str


MIN = -128
MAX = 127


def emit_gemm_data(**kwargs):
    data_str = []
    # Generating matrix size settings
    data_str += [format_scalar_definition("int8_t", "Batch", kwargs["Batch"])]
    data_str += [format_scalar_definition("int8_t", "M", kwargs["M"])]
    data_str += [format_scalar_definition("int8_t", "K", kwargs["K"])]
    data_str += [format_scalar_definition("int8_t", "N", kwargs["N"])]

    # Generating strides settings

    data_str += [
        format_scalar_definition(
            "int32_t", "strideInnermostA", kwargs["strideInnermostA"]
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "strideInnermostB", kwargs["strideInnermostB"]
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "strideInnermostC", kwargs["strideInnermostC"]
        )
    ]

    data_str += [format_scalar_definition("int32_t", "ldA", kwargs["ldA"])]
    data_str += [format_scalar_definition("int32_t", "ldB", kwargs["ldB"])]
    data_str += [format_scalar_definition("int32_t", "ldC", kwargs["ldC"])]

    data_str += [
        format_scalar_definition("int32_t", "strideA", kwargs["strideA"])
    ]
    data_str += [
        format_scalar_definition("int32_t", "strideB", kwargs["strideB"])
    ]
    data_str += [
        format_scalar_definition("int32_t", "strideC", kwargs["strideC"])
    ]

    data_str += [
        format_scalar_definition(
            "int32_t", "delta_local_a", kwargs["delta_local_a"]
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "delta_local_b", kwargs["delta_local_b"]
        )
    ]

    # Generating random 8 integer a and b for subtraction
    subtraction_a = np.random.randint(MIN, MAX)
    subtraction_b = np.random.randint(MIN, MAX)

    # Writing the subtraction value to data.h
    data_str += [
        format_scalar_definition(
            "int8_t", "subtraction_a", subtraction_a
        )
    ]
    data_str += [
        format_scalar_definition(
            "int8_t", "subtraction_b", subtraction_b
        )
    ]

    # Generate random input matrices
    length_a = (
        kwargs["Batch"] * kwargs["M"] * kwargs["K"] * kwargs["meshRow"] * kwargs["tileSize"]
    )
    length_b = (
        kwargs["Batch"] * kwargs["N"] * kwargs["K"] * kwargs["meshCol"] * kwargs["tileSize"]
    )
    I0 = np.random.randint(MIN, MAX, length_a)
    I1 = np.random.randint(MIN, MAX, length_a)
    I2 = np.random.randint(MIN, MAX, length_a)
    I3 = np.random.randint(MIN, MAX, length_a) 
    I4 = np.random.randint(MIN, MAX, length_a)
    I5 = np.random.randint(MIN, MAX, length_a) 

    W0 = np.random.randint(MIN, MAX, length_b)
    W1 = np.random.randint(MIN, MAX, length_b)
    W2 = np.random.randint(MIN, MAX, length_b)
    W3 = np.random.randint(MIN, MAX, length_b)

    # # Generating golden data
    # c0_golden = block_gemm_golden_model(
    #     kwargs["Batch"],
    #     kwargs["M"],
    #     kwargs["K"],
    #     kwargs["N"],
    #     kwargs["meshRow"],
    #     kwargs["tileSize"],
    #     kwargs["meshCol"],
    #     I0,
    #     W0,
    #     subtraction_a,
    #     subtraction_b
    # )
    # c1_golden = block_gemm_golden_model(
    #     kwargs["Batch"],
    #     kwargs["M"],
    #     kwargs["K"],
    #     kwargs["N"],
    #     kwargs["meshRow"],
    #     kwargs["tileSize"],
    #     kwargs["meshCol"],
    #     I0,
    #     W1,
    #     subtraction_a,
    #     subtraction_b
    # )
    # c2_golden = block_gemm_golden_model(
    #     kwargs["Batch"],
    #     kwargs["M"],
    #     kwargs["K"],
    #     kwargs["N"],
    #     kwargs["meshRow"],
    #     kwargs["tileSize"],
    #     kwargs["meshCol"],
    #     I0,
    #     W2,
    #     subtraction_a,
    #     subtraction_b
    # )
    # c3_golden = block_gemm_golden_model(
    #     kwargs["Batch"],
    #     kwargs["M"],
    #     kwargs["K"],
    #     kwargs["N"],
    #     kwargs["meshRow"],
    #     kwargs["tileSize"],
    #     kwargs["meshCol"],
    #     I0,
    #     W3,
    #     subtraction_a,
    #     subtraction_b
    # )

    # res_golden = np.add(c3_golden,I0)

    # Writing testing data and golden data into data.h
    data_str += [format_vector_definition("int8_t", "I0", I0)]
    data_str += [format_vector_definition("int8_t", "I1", I0)]
    data_str += [format_vector_definition("int8_t", "I2", I0)]
    data_str += [format_vector_definition("int8_t", "I3", I0)]
    data_str += [format_vector_definition("int8_t", "I4", I0)]
    data_str += [format_vector_definition("int8_t", "I5", I0)]

    data_str += [format_vector_definition("int8_t", "W0", W0)]
    data_str += [format_vector_definition("int8_t", "W1", W1)]
    data_str += [format_vector_definition("int8_t", "W2", W2)]
    data_str += [format_vector_definition("int8_t", "W3", W3)]
    # data_str += [format_vector_definition("int32_t", "C0_golden", c0_golden)]
    # data_str += [format_vector_definition("int32_t", "C1_golden", c1_golden)]
    # data_str += [format_vector_definition("int32_t", "C2_golden", c2_golden)]
    # data_str += [format_vector_definition("int32_t", "C3_golden", c3_golden)]
    # data_str += [format_vector_definition("int32_t", "res_golden", res_golden)]

    data_str = "\n\n".join(data_str)

    return data_str


def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generate data for kernels")
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