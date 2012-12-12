#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# External Libraries
import re as __re__
import sys  as __sys__
import locale as __locale__

##################################################################
# Constants
COMMENT_RE    = __re__.compile(r'(?:^|\s+)#.*$')
DEFAULT_RE    = __re__.compile(r'(?!)')

##################################################################
# Methods
def skip_comments(istring):
    '''Return input string with comments stripped-off.'''
    return COMMENT_RE.sub('', istring, 1).strip()

##################################################################
# Interface
__all__ = ['lingre', 'lingmap', 'xmap']
