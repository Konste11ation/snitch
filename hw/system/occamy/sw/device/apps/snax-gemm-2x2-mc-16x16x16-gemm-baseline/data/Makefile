# Copyright 2023 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))
CFG_DIR = $(DATA_DIR)/../cfg

CFG_FILE ?= $(CFG_DIR)/gemm_mc_params.hjson

DATA_H = $(DATA_DIR)/data.h

$(DATA_H): $(DATA_DIR)/datagen.py $(CFG_FILE)
	$< -c $(CFG_FILE) > $@

.PHONY: clean-data clean

gen-data: $(DATA_H) FORCE

FORCE: 
clean-data:
	rm -f $(DATA_H)

clean: clean-data