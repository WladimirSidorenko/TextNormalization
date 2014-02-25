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
        split_pos = [end for (start, end) in \
                      split_spans.select_nonintersect(keep_spans)]
        # split input string according to split points
        return self.__split_helper(istring, split_pos)

    def __split_helper(self, istring, splits):
        """Split input string according to split points.

        @return split string and a positions at which newly split sentences
        originally started
        """
        start = 0
        # lead_ws_cnt is the numer of leading whitespaces at the beginning of
        # sentence
        lead_ws_cnt = 0
        output = []
        ret_splits = []
        sentence = ''
        for end in splits:
            sentence, lead_ws_cnt = self.__strip(istring[start:end])
            # prevent empty sentences from being added to the result list
            if sentence:
                ret_splits.append(start + lead_ws_cnt)
                output.append(sentence)
            start = end
        remained, lead_ws_cnt = self.__strip(istring[start:])
        if remained:
            ret_splits.append(start + lead_ws_cnt)
            output.append(remained)
        return output, ret_splits

    def __strip(self, isentence):
        """Remove leading and trailing whitespaces from isentence.

        Remove leading and trailing whitespaces from isentence and return
        modified isentence with the numer of leading white spaces removed.

        """
        lead_ws_cnt = len(isentence)
        # remove leading whitespaces from sentence
        isentence = isentence.lstrip()
        # update `lead_ws_cnt`
        lead_ws_cnt -= len(isentence)
        return isentence.rstrip(), lead_ws_cnt
