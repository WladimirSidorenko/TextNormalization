##################################################################
# Special Variables
# prefer bash over other shells
SHELL := $(or $(shell which bash), /bin/sh)

##################################################################
# Special Targets
.DELETE_ON_ERROR:

##################################################################
# Variables

ifndef SOCMEDIA_ROOT
$(error "Variable SOCMEDIA_ROOT is not set. Run . scipts/set_env first.")
endif

# SOCMEDIA_ROOT comes from version setup script
BIN_DIR  := ${SOCMEDIA_BIN}
DATA_DIR := ${SOCMEDIA_DATA}
TMP_DIR  := ${SOCMEDIA_ROOT}/tmp
DIR_LIST := ${TMP_DIR} ${DATA_DIR}

# Corpus
SRC_CORPUS := ${SOCMEDIA_LSRC}/corpus/twitter_wulff.txt

##################################################################
# PHONY
.PHONY: all help create_dirs character_squeezer \
	topics topics_bernoulli topics_multinomial \
	clean clean_character_squeezer clean_sentiment_tagger \
	clean_topics

##################################################################
# Targets

#################################
# all
all: create_dirs character_squeezer sentiment_tagger

clean: clean_character_squeezer clean_sentiment_tagger \
	clean_topics

#################################
# help
help:
	-@echo -e "This Makefile provides following targets:\n\
	\n\
	all          - build all targets necessary for project\n\
	create_dirs  - create directories for executable files\n\
	character_squeezer   - gather statistics necessary for squeezing\n\
	               duplicated characters\n\
	sentiment_tagger - prepare list of sentiment polarity markers\n\
	topics       - gather statistics necessary for detection of topics\n\
	\n\
	clean        - remove all temporary and binary data\n\
	clean_character_squeezer - remove files created by character_squeezer\n\
	clean_sentiment_tagger - remove lists with sentiment polarity markers\n\
	clean_topics - remove files created by topics\n" >&2

#################################
# create_dirs
create_dirs: ${DIR_LIST}

${DIR_LIST}:
	mkdir -p $@

#################################
# character_squeezer
CHAR_SQUEEZER_CORPUS := ${TMP_DIR}/char_squeezer_corpus.txt
CHAR_SQUEEZER_PICKLE := ${BIN_DIR}/lengthened_stat.pckl

character_squeezer: ${BIN_DIR}/lengthened_stat.pckl | \
		    create_dirs

${CHAR_SQUEEZER_PICKLE}: ${CHAR_SQUEEZER_CORPUS}
	set -e ; \
	lengthened_stat $^ > '${@}.tmp' && mv '${@}.tmp' '$@'

${CHAR_SQUEEZER_CORPUS}: ${SRC_CORPUS}
	set -e -o pipefail; \
	character_normalizer $^ | noise_cleaner -n | \
	umlaut_restorer  > '$@.tmp' && \
	mv '$@.tmp' '$@'

clean_character_squeezer:
	-rm -f ${BIN_DIR}/lengthened_stat.pckl \
	'${CHAR_SQUEEZER_CORPUS}'

#################################
# sentiment_tagger
SENTIMENT_TAGGER_SRCDIR := ${SOCMEDIA_LSRC}/sentiment_dict
SENTIMENT_TAGGER_FILES  := ${DATA_DIR}/negative.txt ${DATA_DIR}/positive.txt

sentiment_tagger: create_dirs ${SENTIMENT_TAGGER_FILES}

# Note, that gawk's functions are locale aware, so setting locale to
# utf-8 will correctly lowercase input letters
${SENTIMENT_TAGGER_FILES}: ${DATA_DIR}/%.txt: ${SENTIMENT_TAGGER_SRCDIR}/%.azm \
					      ${SOCMEDIA_LSRC}/defines.azm
	set -e; \
	test "$${LANG##*\.}" == "UTF-8" && \
	zoem -i "${<}" -o - | \
	gawk -F "\t" -v OFS="\t" '/^##!/{print; next}; NF {sub(/(^|[[:space:]]+)#.*$$/, "")} \
	$$0 {$$1 = tolower($$1); print}' > "${@}.tmp" && \
	mv "${@}.tmp" "${@}"

clean_sentiment_tagger:
	-rm -f ${SENTIMENT_TAGGER_FILES}

#################################
# topics

# number of topics to be distinguished
N_TOPICS := 40
TOPICS_CORPUS := ${TMP_DIR}/topics_corpus.txt
TOPIC_MODEL_PICKLE = ${BIN_DIR}/topics.%.pckl

topics: topics_bernoulli topics_multinomial

topics_bernoulli topics_multinomial: topics_% : ${TOPIC_MODEL_PICKLE}

${TOPIC_MODEL_PICKLE}: ${TOPICS_CORPUS}
	set -e; \
	topics_train_parameters '--model=$*' --number-of-topics=${N_TOPICS} \
	'$<' > '$@.tmp' && mv '$@.tmp' '$@'

${TOPICS_CORPUS}: ${SRC_CORPUS} ${CHAR_SQUEEZER_PICKLE}
	set -e; \
	topics_train_corpus '$<' > '$@.tmp' && \
	mv '${@}.tmp' '$@'

clean_topics:
	-rm -f '${TOPICS_CORPUS}' ${BIN_DIR}/topics*
