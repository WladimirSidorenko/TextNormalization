#####################
# Special Variables #
#####################
# prefer bash over other shells
SHELL := $(or $(shell which bash), /bin/sh)

#############
# Variables #
#############
# `SOCMEDIA_ROOT' is the root directory of this project and should be
# set by executing `. scrips/set_env' prior to running this Makefile
ifndef SOCMEDIA_ROOT
$(error "Variable SOCMEDIA_ROOT is not set. Run . scipts/set_env first.")
endif

# commonly used locations
DATA_DIR := ${SOCMEDIA_DATA}
TMP_DIR  := ${SOCMEDIA_TMP}

# list of automatically created directories (may be extened in firther
# Makefiles)
DIR_LIST := ${TMP_DIR} ${DATA_DIR}

###################
# Special Targets #
###################
.DELETE_ON_ERROR:

.PHONY: all create_dirs all_src all_lingsrc test \
	help help_src help_lingsrc help_test \
	clean_dirs clean_src clean_lingsrc

#####################
# Top-Level Targets #
#####################
# all
all: create_dirs all_src all_lingsrc

all_lingsrc: all_src

all_src all_lingsrc: create_dirs

# clean
clean: clean_dirs clean_src clean_lingsrc
	@MAKE_REPORT=1 ${MAKE} clean_test

############
# Includes #
############
# Makefile with compilation rules for C++ sources
include Makefile.src
# Makefile with compilation rules for linguistic components
include Makefile.lingsrc
# Makefile with rules for testing
include Makefile.test

####################
# Specific Targets #
####################
# create_dirs
create_dirs: ${DIR_LIST}

${DIR_LIST}:
	mkdir -p $@

clean_dirs:
	-rm -rf ${DIR_LIST}

# help
help:
	-@echo -e "### Top-Level Targtes ###\n\
	all          - build all targets necessary for project\n\
	help         - show this screen and exit\n\
	create_dirs  - create auxiliary directories\n\
	\n\
	clean        - remove all created binary and temporary data\n" \
	>&2 && ${MAKE} help_src help_lingsrc help_test > /dev/null
