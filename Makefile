##################################################################
# Special Variables
SHELL := $(or $(wildcard /bin/bash) $(wildcard /usr/pkg/bin/bash))

##################################################################
# Special Targets
.DELETE_ON_ERROR:

##################################################################
# Variables

ifndef SOCMEDIA_ROOT
$(error "Variable SOCMEDIA_ROOT is not set. Run . scipts/set_env first.")
endif

# SOCMEDIA_ROOT comes from version setup script
DIR_LIST := ${SOCMEDIA_BIN} ${SOCMEDIA_TMP}

##################################################################
# PHONY
.PHONY: all create_dirs character_squeezer

##################################################################
# Actual Targets
all: create_dirs character_squeezer

#################################
# create_dirs
create_dirs: ${DIR_LIST}

${DIR_LIST}:
	mkdir -p "$@"

#################################
# character_squeezer
character_squeezer: ${SOCMEDIA_BIN}/lengthened_stat.pckl | \
		    create_dirs

${BIN_DIR}/lengthened_stat.pckl: ${SOCMEDIA_TMP}/corpus.txt
	set -e ; \
	lengthened_stat $^ > "${@}.tmp" && mv "${@}.tmp" "$@"

${TMP_DIR}/corpus.txt : ${SOCMEDIA_LSRC}/corpus/twitter_wulff_text.txt
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
