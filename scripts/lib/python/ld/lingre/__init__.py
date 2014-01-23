#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
import re

##################################################################
# Constants
RE_OPTIONS     = re.compile(r'##!\s*RE_OPTIONS\s*:\s*(\S.*\S)\s*$')
RE_OPTIONS_SEP = re.compile(r'\s*\|\s*')

##################################################################
# Interface
__all__ = ['lre', 'lre_match']
