#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Description
"""Module for handling information about token offsets.

   Constants:
   OFFSET_TAG_FMT - format string for printing information about offsets
   OFFSET_TAG_RE - regular expression for parsing information about offsets

   Classes:
   Offsets - class for storing and modifying information about
             token offsets

"""

##################################################################
# Modules
import os
import re
import sys
from collections import deque

##################################################################
# Constants
START_LEN_SEP = "::"
__OFFSET_TAG_XXX__ = os.environ.get("SOCMEDIA_ESC_CHAR", "") + \
    "\ttoken_offsets\tsentence\t{:s}\toffsets{:s}"
OFFSET_TAG_FMT = unicode(__OFFSET_TAG_XXX__.format(r"{:d}", "\t{:s}"))
OFFSET_TAG_RE  = re.compile(__OFFSET_TAG_XXX__.format(r"(\d+)", r"((?:\t\d+)+)"))

##################################################################
# Class
class Offsets:
    """
    This class provides methods and structures for storing and updating
    information about token offsets.

    Instance Variables:
    self.sent_cnt - counter of sentences
    self.sentences - list of offsets for individual sentences

    Methods:
    __init__() - initialize instance variables
    clear() - clear all information about offsets
    is_empty() - boolean function indicating whether offset container is empty
    popleft_sentence() - return offset list for the 1-st sentence in queue
    popleft_token() - return offset and length of the 1-st token in queue
    parse() - parse input line with meta information about offsets
    __str__() - return string representation of offsets

    """

    def __init__(self):
        """Initialize instance variables."""
        self.sent_cnt = 0
        self.sentences = deque([])

    def popleft_sentence(self):
        """Return offset list for the 1-st sentence in the queue."""
        if self.sent_cnt > 0:
            self.sent_cnt -= 1
            return self.sentences.popleft()
        else:
            return []

    def popleft_token(self):
        """Return offset and length of the 1-st word in queue.

        @return 2-tuple with first element representing the offset and second
        element representing the length of the first token in queue.  Tuple
        (None, None) will be returned if no elements are left.

        """
        if self.sent_cnt > 0:
            if self.sentences[0]:
                return self.sentences[0].pop(0)
            else:
                self.popleft_sentence()
                self.popleft_token()
        else:
            return (None, None)

    def append(self, t_list):
        """Append list of token offsets to the end of internal storage.

        @param o_list - list of 2-tuples representing whose first element is
        information about token starting position, and second element

        """
        self.sent_cnt += 1
        self.sentences.append(t_list)

    def is_empty(self):
        """Boolean function indicating whether any information about offsets is present."""
        return self.sent_cnt == 0

    def parse(self, iline):
        """Parse input line with meta information about offsets.

        @param iline - input line containing meta information

        @return true if the line could be parsed correctly, false otherwise

        """
        mobj = OFFSET_TAG_RE.match(iline)
        # if line does not match format, skip it
        if not mobj:
            return False
        # otherwise, determine sentence number and accommodate the offset list
        # at appropriate place
        s_cnt = mobj.group(1)
        t_offsets = [ofs.split(START_LEN_SEP) for ofs in '\t'.split(mobj.group(2)) if ofs]
        t_offsets = [(int(t_start), int(t_len)) for t_start, t_len in t_offsets]
        # if we have alread seen that many sentences, than overwrite the
        # information
        if s_cnt < self.sent_cnt:
            self.sentences[s_cnt] = t_offsets
        # otherwise, append as many new token containers as needed to store
        # this list of token offsets
        else:
            for i in xrange(self.sent_cnt + 1, s_cnt):
                self.sentences.append([])
            self.sent_cnt = s_cnt
            self.sentences[s_cnt - 1].extend(t_offsets)

    def clear(self):
        """Remove all information about offsets."""
        self.sent_cnt = 0
        self.sentences.clear()

    def __str__(self):
        """Return string representation of the offsets.

        @return string representing meta-information about offsets

        """
        ret = []
        ofs_line = ''
        offsets = ''
        for s_cnt, snt in enumerate(self.sentences):
            offsets = '\t'.join([str(start) + START_LEN_SEP + str(length) \
                                     for (start, length) in snt])
            ret.append(OFFSET_TAG_FMT.format(s_cnt, offsets))
        return '\n'.join(ret)
