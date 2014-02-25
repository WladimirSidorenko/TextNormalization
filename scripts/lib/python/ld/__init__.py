#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Interface
__all__ = ["character_squeezer", "lingre", "lingmap", "misspellings", \
               "noise_restorer", "p2p", "repeated_chars", "stringtools", \
               "umlauts"]

##################################################################
# External Libraries
import os
import re
import sys as __sys__
import string as __string__
import locale as __locale__

##################################################################
# Constants
W_SEP      = re.compile(r"[\s{}\\]+".format(__string__.punctuation), \
                                re.UNICODE)
COMMENT_RE = re.compile(r"(?:^|\s+)#.*$")
DEFAULT_RE = re.compile(r"(?!)")
# default n-gram files
UNIGRAM_DEFAULT_FILE = "{SOCMEDIA_LINGBIN}/unigram_stat.pckl".format(**os.environ)
BIGRAM_DEFAULT_FILE  = "{SOCMEDIA_LINGBIN}/bigram_stat.pckl".format(**os.environ)

##################################################################
# Methods
def skip_comments(istring):
    """Return input string with comments stripped-off."""
    return COMMENT_RE.sub('', istring, 1).strip()

##################################################################
# Exceptions
class RuleFormatError(Exception):
    def __init__(self, msg = "", efile = None):
        msg = msg.encode("utf-8")
        if not efile:
            Exception.__init__(self, msg)
        else:
            line = efile.line.encode("utf-8")
            Exception.__init__(self, """
The following rule line could not be parsed correctly:
File:        '{0.filename:s}'
Line #:      {0.fnr:d}
Line:        '{1!s:s}'
Repr:        {1!r:s}""".format(efile, line) + '\n' + msg)
