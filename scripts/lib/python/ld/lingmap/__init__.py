#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
from .. import __re__

##################################################################
# Constants
MAP_DELIMITER = __re__.compile(r'(?<=[^\\])\t+ *')

##################################################################
# Interface
__all__ = ['lmap']
