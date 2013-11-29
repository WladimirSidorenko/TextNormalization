#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Utility for determining sentence boundaries.
"""

##################################################################
# Libraries
import os
import sys

from ld.lingre import lre
from ld.stringtools import is_xml_tag

##################################################################
# Constants
RULE_DIR   = "{SOCMEDIA_LINGSRC}/sentence_splitter/".format(**os.environ)
KEEP_FILE  = RULE_DIR + "keep.re"
SPLIT_FILE = RULE_DIR + "divide.re"

##################################################################
# Class
class SentenceSplitter:
    """Class used for determining sentence boundaries."""

    def __init__(self, keep_file = "", split_file = "", verbose = False):
        """Create an instance of SentenceSplitter."""
        self.keep_re  = lre.MultiRegExp(keep_file if keep_file else KEEP_FILE)
        self.split_re = lre.MultiRegExp(split_file if split_file else SPLIT_FILE)
        self.verbose = verbose

    def split(self, istring):
        """Public method for splitting strings."""
        # Calculate regexp spans for all regular expressions from
        # split_re set and keep_re set which matched the input
        # string.
        split_spans = self.split_re.finditer(istring)
        keep_spans  = self.keep_re.finditer(istring)
        if self.verbose:
            print >> sys.stderr, "Split spans are", split_spans
            print >> sys.stderr, "Keep spans are",  keep_spans
        # Filter-out those split spans which intersect with keep spans
        # and remember only the end points of the split spans left
        # over.
        splits = [end for (start, end) in \
                      split_spans.select_nonintersect(keep_spans)]
        # split input string according to split points
        return self._split_helper(istring, splits)

    def _split_helper(self, istring, splits):
        """Split input string according to split points."""
        start = 0
        output = []
        sentence = ''
        for end in splits:
            sentence = istring[start:end].strip()
            # prevent empty sentences from being added to result list
            if sentence:
                output.append(sentence)
            start = end
        remained = istring[start:].strip()
        if remained:
            output.append(remained)
        return output

