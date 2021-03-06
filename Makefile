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
SCRIPT_DIR := ${SOCMEDIA_ROOT}/scripts
# list of automatically created directories (may be extened in firther
# Makefiles)
DIR_LIST := ${TMP_DIR} ${DATA_DIR} ${SOCMEDIA_TTAGGER_DIR} ${SOCMEDIA_MPARSER_DIR} \
	 ${SOCMEDIA_SRL_DIR} ${SOCMEDIA_CRF_DIR} ${SOCMEDIA_SEMDICT_DIR}

###################
# Special Targets #
###################
.DELETE_ON_ERROR:

.PHONY: all \
	create_dirs \
	fetch fetch_tagger fetch_parser fetch_srl \
	fetch_alchemy fetch_crf fetch_weka \
	all_src all_lingsrc \
	test \
	help help_src help_lingsrc help_test \
	clean_dirs \
	clean_fetch clean_fetch_tagger clean_fetch_parser clean_fetch_srl \
	clean_fetch_alchemy clean_fetch_crf clean_fetch_weka \
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
	fetch_srl    - download MateSematicRoleLabeler\n\
	fetch_alchemy - download alchemy-2 package\n\
	fetch_crf    - download CRFSuite\n\
	fetch_weka    - download LibWEKA\n\
	\n\
	clean        - remove all created binaries and temporary data\n\
	clean_fetch  - remove all 3-rd parties software\n\
	clean_fetch_tagger - remove TreeTagger\n\
	clean_fetch_parser - remove MateParser\n\
	clean_fetch_srl    - remove MateSematicRoleLabeler\n\
	clean_fetch_alchemy - remove Alchemy-2 files\n\
	clean_fetch_crf - remove CRFSuite\n\
	clean_fetch_weka - remove LibWEKA\n\
	"  >&2 && ${MAKE} help_src help_lingsrc help_test > /dev/null

############
# Includes #
############
# since includes might extend or modify variables which are used
# further in this file, we need to put them here

# Makefile with compilation rules for C++ sources
# include Makefile.src
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
	(umask 0002 && mkdir -p $@;)

clean_dirs:
	-rm -rf ${DIR_LIST}

#############################
# fetch external dependencies
fetch: fetch_tagger fetch_parser fetch_srl \
	fetch_crf fetch_weka  # fetch_alchemy

clean_fetch: clean_fetch_tagger clean_fetch_parser clean_fetch_srl \
	clean_fetch_alchemy clean_fetch_crf clean_fetch_weka

###################
# fetch tree-tagger
TTAGGER_HTTP_ADDRESS := 'https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data'
TTAGGER_BIN_FILE   := ${SOCMEDIA_TTAGGER_DIR}/bin/tree-tagger
TTAGGER_PARAM_FILE := ${SOCMEDIA_TTAGGER_DIR}/german.par

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
# fetch Mate Parser
MTOOLS_HTTP_ADDRESS := https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/mate-tools

MPARSER_JAR_FILE := ${SOCMEDIA_MPARSER_DIR}/anna-3.61.jar
MPARSER_MODEL := ${SOCMEDIA_MPARSER_DIR}/ger-tagger+lemmatizer+morphology+graph-based-3.6+.tgz
MPARSER_MODEL_ADDRESS := $(MTOOLS_HTTP_ADDRESS)/$(subst +,%2B,$(notdir $(MPARSER_MODEL)))
MPARSER_PARSE_MODEL := ${SOCMEDIA_MPARSER_DIR}/parser-ger-3.6.model
MPARSER_MTAGGER_MODEL := ${SOCMEDIA_MPARSER_DIR}/morphology-ger-3.6.model

fetch_parser: ${MPARSER_JAR_FILE} ${MPARSER_PARSE_MODEL} \
	${MPARSER_MTAGGER_MODEL}

${MPARSER_JAR_FILE}:
	set -e; cd ${@D} && wget '${MTOOLS_HTTP_ADDRESS}/${@F}'

${MPARSER_MODEL}:
	set -e; cd ${@D} && wget '${MPARSER_MODEL_ADDRESS}'

${MPARSER_PARSE_MODEL} ${MPARSER_MTAGGER_MODEL}: ${MPARSER_MODEL}
	set -e -o pipefail; \
	cd ${@D} && tmp_file="$$(tar --wildcards -tzf `basename $<` '*/${@F}')" && \
	tar -xzf "$$(basename $<)" "$${tmp_file}" && mv "$${tmp_file}" ${@F}

clean_fetch_parser:
	-rm -rf ${MPARSER_JAR_FILE} ${MPARSER_MODEL} \
	${MPARSER_PARSE_MODEL} ${MPARSER_MTAGGER_MODEL}

#################################
# fetch Mate Sematic Role Labeler
SRL_ARCHIVE := ${SOCMEDIA_SRL_DIR}/srl-4.31.tgz
SRL_HTTP_ADDRESS := ${MTOOLS_HTTP_ADDRESS}/$(notdir $(SRL_ARCHIVE))

SRL_JAR_FILE := ${SOCMEDIA_SRL_DIR}/srl.jar
SRL_LIBLINEAR_FILE := ${SOCMEDIA_SRL_DIR}/liblinear-1.51-with-deps.jar
SRL_ANNA_FILE := ${SOCMEDIA_SRL_DIR}/anna-3.3.jar

SRL_MODEL := ${SOCMEDIA_SRL_DIR}/tiger-complete-predsonly-srl-4.11.srl.model
SRL_MODEL_ADDRESS := $(MTOOLS_HTTP_ADDRESS)/$(notdir $(SRL_MODEL))

fetch_srl: ${SRL_JAR_FILE} ${SRL_LIBLINEAR_FILE} ${SRL_ANNA_FILE} ${SRL_MODEL}

${SRL_ARCHIVE}:
	set -e; cd ${@D} && wget '${SRL_HTTP_ADDRESS}'

${SRL_JAR_FILE} ${SRL_LIBLINEAR_FILE} ${SRL_ANNA_FILE}: ${SRL_ARCHIVE}
	set -e -o pipefail; \
	cd ${@D} && tmp_file="$$(tar --wildcards -mtzf '$<' '*/${@F}')" && \
	tar -mxzf '$<' "$${tmp_file}" && mv "$${tmp_file}" ${@F} && \
	rm -r "$$(basename $${tmp_file})"

${SRL_MODEL}:
	set -e; cd ${@D} && wget '${SRL_MODEL_ADDRESS}'

clean_fetch_srl:
	-rm -rf ${SRL_ARCHIVE} ${SRL_JAR_FILE} ${SRL_LIBLINEAR_FILE} \
	${SRL_ANNA_FILE} ${SRL_MODEL}

###################
# fetch alchemy
ALCHEMY_HTTP_ADDRESS := https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/alchemy-2/alchemy-2.tar.gz
ALCHEMY_MAKEFILE  := ${SOCMEDIA_ALCHEMY_DIR}/src/makefile
ALCHEMY_BINDIR := ${SOCMEDIA_ALCHEMY_DIR}/bin
ALCHEMY_BIN_FILES := $(addprefix $(ALCHEMY_BINDIR)/, learnwts learnstruct liftedinfer runliftedinfertests)

# `ALCHEMY_MAKEFILE` will describe how to make these binary files
fetch_alchemy: ${ALCHEMY_BIN_FILES}

${ALCHEMY_BIN_FILES}: ${ALCHEMY_MAKEFILE}
	set -e; \
	${MAKE} GPP=g++ -C ${<D} -f ${<F} ${@F}

${ALCHEMY_MAKEFILE}:
	set -e; \
	cd $(dir $(SOCMEDIA_ALCHEMY_DIR)) && \
	wget '${ALCHEMY_HTTP_ADDRESS}' && \
	tar -xzf $(notdir $(ALCHEMY_HTTP_ADDRESS)) $(notdir $(SOCMEDIA_ALCHEMY_DIR)) && \
	rm -f $(notdir $(ALCHEMY_HTTP_ADDRESS))

clean_fetch_alchemy:
	-rm -rf ${SOCMEDIA_ALCHEMY_DIR}

###################
# fetch CRFSuite
CRF_BIN_FILE := ${SOCMEDIA_CRF_DIR}/crfsuite
CRF_HTTP_ADDRESS := https://github.com/downloads/chokkan/crfsuite/crfsuite-0.12-x86_64.tar.gz

fetch_crf: ${CRF_BIN_FILE}

${CRF_BIN_FILE}:
	set -e -o pipefail; \
	cd ${SOCMEDIA_CRF_DIR} && \
	wget '${CRF_HTTP_ADDRESS}' && \
	tar --wildcards -xzf '$(notdir $(CRF_HTTP_ADDRESS))' 'crfsuite-*/bin/crfsuite' && \
	mv crfsuite-*/bin/crfsuite $@ && \
	rm -rf crfsuite-* '$(notdir $(CRF_HTTP_ADDRESS))'

clean_fetch_crf:
	-rm -rf ${CRF_BIN_FILE}

###################
# fetch LibWEKA
WEKA_HTTP_ADDRESS := http://prdownloads.sourceforge.net/weka/weka-3-6-10.zip
WLSVM_HTTP_ADDRESS := http://www.cs.iastate.edu/~yasser/wlsvm/wlsvm.zip

WEKA_LIB := ${SOCMEDIA_WEKA_DIR}/weka.jar
LSVM_LIBS := ${SOCMEDIA_WEKA_DIR}/libsvm.jar ${SOCMEDIA_WEKA_DIR}/wlsvm.jar

fetch_weka: ${WEKA_LIB}
# fetch_weka: ${WEKA_LIB}  ${LSVM_LIBS}

${WEKA_LIB}:
	set -e -o pipefail; \
	cd $(dir $(SOCMEDIA_WEKA_DIR)); \
	wget '${WEKA_HTTP_ADDRESS}' && \
	unzip -o '$(notdir $(WEKA_HTTP_ADDRESS))' && \
	(test '$(notdir $(basename $(WEKA_HTTP_ADDRESS)))' != '$(notdir $(SOCMEDIA_WEKA_DIR))' && \
	echo 'HTTP archive name does not match target directory for WEKA.' >&2 && exit 1;) || \
	rm -f '$(notdir $(WEKA_HTTP_ADDRESS))'

# ${LSVM_LIBS}:
# 	set -e; \
# 	cd '$(SOCMEDIA_WEKA_DIR)' && wget '${WLSVM_HTTP_ADDRESS}' && \
# 	unzip -o '$(notdir $(WLSVM_HTTP_ADDRESS))' && \
# 	mv 'WLSVM/lib/${@F}' $@ && \
# 	rm -rf $(notdir $(WLSVM_HTTP_ADDRESS)) WLSVM/

clean_fetch_weka:
	-rm -rf $(wildcard $(SOCMEDIA_WEKA_DIR)/*)
