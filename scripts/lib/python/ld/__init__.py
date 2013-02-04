#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Interface
__all__ = ['lingre', 'lingmap', 'p2p', 'repeated_chars', \
               'stringtools']

##################################################################
# External Libraries
import re  as __re__
import sys as __sys__
import string as __string__
import locale as __locale__

##################################################################
# Constants
W_SEP      = __re__.compile(r'[\s{}\\]+'.format(__string__.punctuation), \
                                __re__.UNICODE)
COMMENT_RE = __re__.compile(r'(?:^|\s+)#.*$')
DEFAULT_RE = __re__.compile(r'(?!)')

##################################################################
# Methods
def skip_comments(istring):
    '''Return input string with comments stripped-off.'''
    return COMMENT_RE.sub('', istring, 1).strip()

##################################################################
# Exceptions
class RuleFormatError(Exception):
    def __init__(self, msg = '', efile = None):
        msg = msg.encode('utf-8')
        line = efile.line.encode('utf-8')
        if not efile:
            Exception.__init__(self, msg)
        else:
            Exception.__init__(self, """
The following rule line could not be parsed correctly:
File:        '{0.filename:s}'
Line #:      {0.fnr:d}
Line:        '{1!s:s}'
Repr:        {1!r:s}""".format(efile, line) + "\n" + msg)
