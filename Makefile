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
TMP_DIR   := ${SOCMEDIA_TMP}

# list of automatically created directories (may be extened in firther
# Makefiles)
DIR_LIST := ${TMP_DIR} ${DATA_DIR} ${SOCMEDIA_TTAGGER_DIR} ${SOCMEDIA_MPARSER_DIR}

###################
# Special Targets #
###################
.DELETE_ON_ERROR:

.PHONY: all \
	create_dirs \
	fetch fetch_tagger fetch_parser fetch_alchemy \
	all_src all_lingsrc \
	test \
	help help_src help_lingsrc help_test \
	clean_dirs \
	clean_fetch clean_fetch_tagger clean_fetch_parser clean_fetch_alchemy \
	clean_src clean_lingsrc

#####################
# Top-Level Targets #
#####################

#######
# all
all: create_dirs fetch all_src all_lingsrc

all_lingsrc: all_src

all_src all_lingsrc: create_dirs fetch

#######
# clean
clean: clean_dirs clean_fetch clean_src clean_lingsrc
	@MAKE_REPORT=1 ${MAKE} clean_test

#######
# help
help:
	-@echo -e "### Top-Level Targtes ###\n\
	all          - build all targets needed for project\n\
	help         - show this screen and exit\n\
	create_dirs  - create directories needed for storing compiled files\n\
	fetch        - download all needed 3-rd parties software\n\
	fetch_tagger - download TreeTagger\n\
	fetch_parser - download MateParser\n\
	fetch_alchemy - download alchemy-2 package\n\
	\n\
	clean        - remove all created binaries and temporary data\n\
	clean_fetch  - remove all 3-rd parties software\n\
	clean_fetch_tagger - remove TreeTagger\n\
	clean_fetch_parser - remove MateParser\n\
	clean_fetch_alchemy - remove Alchemy-2 files\n\
	"  >&2 && ${MAKE} help_src help_lingsrc help_test > /dev/null

############
# Includes #
############
# since includes might extend or modify variables which are used
# further in this file, we need to put them here

# Makefile with compilation rules for C++ sources
include Makefile.src
# Makefile with compilation rules for linguistic components
include Makefile.lingsrc
# Makefile with rules for testing
include Makefile.test

####################
# Specific Targets #
####################

#############
# create_dirs
create_dirs: ${DIR_LIST}

${DIR_LIST}:
	mkdir -p $@

clean_dirs:
	-rm -rf ${DIR_LIST}

#############################
# fetch external dependencies
fetch: fetch_tagger fetch_parser fetch_alchemy

clean_fetch: clean_fetch_tagger clean_fetch_parser clean_fetch_alchemy

###################
# fetch tree-tagger
TTAGGER_HTTP_ADDRESS := 'http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data'
TTAGGER_BIN_FILE   := ${SOCMEDIA_TTAGGER_DIR}/bin/tree-tagger
TTAGGER_PARAM_FILE := ${SOCMEDIA_TTAGGER_DIR}/german-par-linux-3.2-utf8.bin

fetch_tagger: ${TTAGGER_BIN_FILE} ${TTAGGER_PARAM_FILE}

ifeq "" "Darwin"
${TTAGGER_BIN_FILE}: TTAGGER_DOWNLOAD_FILE := ${TTAGGER_HTTP_ADDRESS}/tree-tagger-MacOSX-3.2-intel.tar.gz
else
${TTAGGER_BIN_FILE}: TTAGGER_DOWNLOAD_FILE := ${TTAGGER_HTTP_ADDRESS}/tree-tagger-linux-3.2.tar.gz
endif

${TTAGGER_BIN_FILE}:
	set -e; \
	cd ${SOCMEDIA_TTAGGER_DIR} && \
	wget '${TTAGGER_DOWNLOAD_FILE}' && \
	tar -xzf $(notdir $(TTAGGER_DOWNLOAD_FILE)) bin/tree-tagger && \
	rm -f $(notdir $(TTAGGER_DOWNLOAD_FILE))

${TTAGGER_PARAM_FILE}:
	set -e; \
	cd ${@D} && wget '${TTAGGER_HTTP_ADDRESS}/${@F}.gz' && \
	gzip -d ${@}.gz && rm -f ${@}.gz

clean_fetch_tagger:
	-rm -rf ${TTAGGER_BIN_FILE} ${TTAGGER_PARAM_FILE}

###################
# fetch mate-parser
MPARSER_HTTP_ADDRESS := 'http://mate-tools.googlecode.com/files'
MPARSER_JAR_FILE := ${SOCMEDIA_MPARSER_DIR}/anna-3.3.jar
MPARSER_PARSE_MODEL_FILE := ${SOCMEDIA_MPARSER_DIR}/tiger-complete.anna-3-1.parser.model
MPARSER_MTAGGER_MODEL_FILE := ${SOCMEDIA_MPARSER_DIR}/tiger-complete.anna-3-1.morphtagger.model

fetch_parser: ${MPARSER_JAR_FILE} \
	${MPARSER_PARSE_MODEL_FILE} \
	${MPARSER_MTAGGER_MODEL_FILE}

${MPARSER_JAR_FILE} ${MPARSER_PARSE_MODEL_FILE} \
	${MPARSER_MTAGGER_MODEL_FILE}:
	set -e; \
	cd ${@D} && wget '${MPARSER_HTTP_ADDRESS}/${@F}'

clean_fetch_parser:
	-rm -rf ${MPARSER_JAR_FILE} ${MPARSER_PARSE_MODEL_FILE} \
	${MPARSER_MTAGGER_MODEL_FILE}

###################
# fetch alchemy
ALCHEMY_HTTP_ADDRESS := http://alchemy-2.googlecode.com/files/alchemy-2.tar.gz
ALCHEMY_MAKEFILE  := ${SOCMEDIA_ALCHEMY_DIR}/src/makefile
ALCHEMY_BINDIR := ${SOCMEDIA_ALCHEMY_DIR}/bin
ALCHEMY_BIN := $(addprefix $(ALCHEMY_BINDIR)/, learnwts learnstruct liftedinfer runliftedinfertests)

# `ALCHEMY_MAKEFILE` will describe how to make these binary files
fetch_alchemy: ${ALCHEMY_BIN}

${ALCHEMY_MAKEFILE}:
	set -e; \
	cd $(dir $(SOCMEDIA_ALCHEMY_DIR)) && \
	wget '${ALCHEMY_HTTP_ADDRESS}' && \
	tar -xzf $(notdir $(ALCHEMY_HTTP_ADDRESS)) $(notdir $(SOCMEDIA_ALCHEMY_DIR)) && \
	rm -f $(notdir $(ALCHEMY_HTTP_ADDRESS))

${ALCHEMY_BIN}: ${ALCHEMY_MAKEFILE}
	set -e; \
	${MAKE} -C ${SOCMEDIA_ALCHEMY_DIR}/src/ ${@F}

clean_fetch_alchemy:
	-rm -rf ${SOCMEDIA_ALCHEMY_DIR}
