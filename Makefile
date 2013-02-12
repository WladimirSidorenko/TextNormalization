##################################################################
# Special Variables
SHELL := $(shell which bash)

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
TMP_DIR  := ${SOCMEDIA_ROOT}/tmp
DIR_LIST := ${BIN_DIR} ${TMP_DIR}

##################################################################
# PHONY
.PHONY: all create_dirs character_squeezer \
	clean clean_character_squeezer

##################################################################
# Actual Targets
all: create_dirs character_squeezer

clean: clean_character_squeezer

#################################
# create_dirs
create_dirs: ${DIR_LIST}

${DIR_LIST}:
	mkdir -p "$@"

#################################
# character_squeezer
character_squeezer: ${BIN_DIR}/lengthened_stat.pckl | \
		    create_dirs

${BIN_DIR}/lengthened_stat.pckl: ${TMP_DIR}/corpus.txt
	set -e ; \
	lengthened_stat $^ > "${@}.tmp" && mv "${@}.tmp" "$@"

${TMP_DIR}/corpus.txt : ${SOCMEDIA_LSRC}/corpus/twitter_wulff.txt
	set -e -o pipefail; \
	character_normalizer -m \
	"${SOCMEDIA_LSRC}/character_normalizer/char2char.map" $^ | \
	noise_cleaner -n -m \
	"${SOCMEDIA_LSRC}/noise_cleaner/noise_cleaner.p2p" | \
	umlaut_restorer	 \
	-r "${SOCMEDIA_LSRC}/umlaut_restorer/misspelled_umlaut.re" \
	-m "${SOCMEDIA_LSRC}/umlaut_restorer/misspelled2umlaut.map" \
	-e "${SOCMEDIA_LSRC}/umlaut_restorer/umlaut_exceptions.dic" > "$@.tmp" && \
	mv "$@.tmp" "$@"

clean_character_squeezer:
	-rm -f ${BIN_DIR}/lengthened_stat.pckl \
	${SOCMEDIA_LSRC}/corpus/twitter_wulff.txt
