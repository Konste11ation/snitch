# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# Add user applications to APPS variable
APPS = offload
APPS += axpy
APPS += dma-2d-test
APPS += snax-mac
APPS += snax-mac-mc
APPS += snax-gemm
APPS += snax-gemm-test
APPS += snax-gemm-test-mc
APPS += snax-gemm-2x2-mc-16x16x16-gemm
APPS += snax-gemm-2x2-mc-16x16x16-gemm-baseline

APPS += snax-gemm-2x2-mc-8x8x8-gemm-residue
APPS += snax-gemm-2x2-mc-8x8x8-gemm-residue-baseline
APPS += snax-gemm-2x2-mc-noc
APPS += snax-gemm-2x2-mc-xbar
APPS += snax-gemm-4x4-mc-32x32x32-gemm-baseline
APPS += snax-gemm-4x4-mc-32x32x32-gemm
TARGET ?= all

APP_SUBDIRS = $(addprefix apps/,$(APPS))

SUBDIRS  = runtime snax/gemm snax/mac
SUBDIRS += $(APP_SUBDIRS)

.PHONY: all $(SUBDIRS)

all: $(SUBDIRS)

runtime:
	$(MAKE) -C $@ $(TARGET)

snax/gemm:
	$(MAKE) -C $@ $(TARGET)

snax/mac:
	$(MAKE) -C $@ $(TARGET)


# Explicit dependency of apps on runtime
$(APP_SUBDIRS): runtime snax/mac snax/gemm
	$(MAKE) -C $@ $(TARGET)
