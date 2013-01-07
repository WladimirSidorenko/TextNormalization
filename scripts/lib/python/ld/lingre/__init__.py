#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
from .. import __re__

##################################################################
# Constants
RE_OPTIONS     = __re__.compile(r'^##!\s*RE_OPTIONS\s*:\s*(\S.*\S)\s*$')
RE_OPTIONS_SEP = __re__.compile(r'\s*|\s*')

##################################################################
# Interface
__all__ = ['lre', 'lre_match']
