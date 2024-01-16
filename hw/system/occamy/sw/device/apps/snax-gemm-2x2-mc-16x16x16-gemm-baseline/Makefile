# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Ryan Antonio <ryan.antonio@esat.kuleuven.be>

APP     = snax-gemm-2x2-mc-16x16x16-gemm-baseline
SRCS    = src/snax-gemm-2x2-mc-16x16x16-gemm-baseline.c
INCDIRS = data
INCDIRS += ../../snax/gemm/include
RISCV_LDFLAGS += ../../snax/gemm/build/snax-gemm-lib.o

include ../common.mk
