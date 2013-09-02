#!/usr/bin/env python2.7

##################################################################
# Libraries
from . import __re__

##################################################################
# Constants
RE_FLAGS = __re__.IGNORECASE | __re__.UNICODE
LEGAL_REPETITION_RE = __re__.compile(r'^(?:"?@.+|[VILXCD]+|[$\d.,]+)\.?$')
REPEATED_LETTERS_RE = __re__.compile(r'([^\W\d_])(\1+)', RE_FLAGS)
THREE_LETTERS_RE    = __re__.compile(r'([^\W\d_])(\1{2})', RE_FLAGS)
TWO_LETTERS_RE    = __re__.compile(r'([^\W\d_])(\1)', RE_FLAGS)
ONE_REPEATED_LETTER = __re__.compile(r'([^\W\d_])(\1+)$', RE_FLAGS)
GT_THREE_LETTERS_RE = __re__.compile(r'(\w)(\1{2})(\1+)', RE_FLAGS)

##################################################################
# Methods
def squeeze(iword):
    '''Replace multiple consecutive occurrences of characters by
    just one.'''
    return REPEATED_LETTERS_RE.sub(r'\1', iword)

# create an alias for squeeze method
generate_key = squeeze
