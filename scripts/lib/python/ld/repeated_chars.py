#!/usr/bin/env python2.7

##################################################################
# Libraries
import re

##################################################################
# Constants
RE_FLAGS = re.IGNORECASE | re.UNICODE
LEGAL_REPETITION_RE = re.compile(r'^(?:"?@.+|[VILXCD]+|[$\d.,]+)\.?$')
REPEATED_LETTERS_RE = re.compile(r'([^\W\d_])(\1+)', RE_FLAGS)
THREE_LETTERS_RE    = re.compile(r'([^\W\d_])(\1{2})', RE_FLAGS)
TWO_LETTERS_RE      = re.compile(r'([^\W\d_])(\1)', RE_FLAGS)
ONE_REPEATED_LETTER = re.compile(r'([^\W\d_])(\1+)$', RE_FLAGS)
GT_THREE_LETTERS_RE = re.compile(r'(\w)(\1{2})(\1+)', RE_FLAGS)

##################################################################
# Methods
def squeeze(iword):
    '''Replace multiple consecutive occurrences of characters by
    just one.'''
    return REPEATED_LETTERS_RE.sub(r'\1', iword)

# create an alias for squeeze method
generate_key = squeeze
