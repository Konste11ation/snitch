# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SRC_DIR := $(realpath $(MK_DIR))
CFG_DIR = $(SRC_DIR)/../cfg
CODE_CFG ?= $(CFG_DIR)/gemm_mc_params.hjson

CODE = $(SRC_DIR)/snax-gemm-4x4-mc-32x32x32-gemm.c

$(CODE): $(SRC_DIR)/gemmgen.py $(CODE_CFG)
	$< -c $(CODE_CFG) > $@

.PHONY: clean-code clean

gen-code: $(CODE) FORCE
FORCE: 

clean-code:
	rm -f $(CODE)

clean: clean-code