# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# Add user applications to APPS variable
APPS  = hello_world
APPS += offload
APPS += axpy
APPS += snax-mac
APPS += snax-mac-mc
#APPS += snax-mac-simple/untiled
#APPS += snax-mac-simple/tiled
TARGET ?= all

SUBDIRS = $(addprefix apps/,$(APPS))

.PHONY: all $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
