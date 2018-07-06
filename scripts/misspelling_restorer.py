#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Script for correcting colloquial writings.

"""

##################################################################
# Libraries
from alt_argparse import argparser
from alt_fio import AltFileInput, AltFileOutput
from ld import UNIGRAM_DEFAULT_FILE, BIGRAM_DEFAULT_FILE
from ld.misspellings import MisspellingRestorer
from replacements import Memory

import os
import pickle

##################################################################
# Constants

##################################################################
# Variables


##################################################################
# Methods
def print_mem():
    """Output and erase memory information if it is not empty."""
    if not memory.is_empty():
        foutput.fprint(unicode(memory))
        memory.forget_all()


##################################################################
# Processing Arguments
# note: some options are already set up by alt_argparser
argparser.description = "Utility for correcting colloquial spellings in text"
argparser.add_argument("-c", "--esc-char",
                       help="escape character which should precede lines"
                       " with meta-information", nargs=1, type=str,
                       default=os.environ.get("SOCMEDIA_ESC_CHAR", ""))
argparser.add_file_argument("-b", "--bigram-prob-file",
                            help="file with bigram probabilities",
                            default=BIGRAM_DEFAULT_FILE)
argparser.add_file_argument("-u", "--unigram-prob-file",
                            help="file with unigram probabilities",
                            default=UNIGRAM_DEFAULT_FILE)
argparser.add_argument("-v", "--verbose",
                       help="switch verbose statistics mode on",
                       action="store_true")
args = argparser.parse_args()


##################################################################
# Main
unigram_prob = pickle.load(args.unigram_prob_file)
args.unigram_prob_file.close()
bigram_prob = pickle.load(args.bigram_prob_file)
args.bigram_prob_file.close()

esc_char = args.esc_char
skip_line = args.skip_line

foutput = AltFileOutput(encoding=args.encoding, flush=args.flush)
finput = AltFileInput(*args.files, print_func=foutput.fprint,
                      errors='replace')
memory = Memory()

# unfortunately, rules for restoration of misspellings are currently hard-coded
# in the `misspellings.py` file
misspelling_restorer = MisspellingRestorer(unigram_prob, bigram_prob)

# iterate over input lines, skip empty and skip lines, pre-cache information
# about replacements
for line in finput:
    # print empty and skip lines unchanged
    if line == skip_line or not line:
        # check if memory is empty and print it otherwise
        print_mem()
        foutput.fprint(line)
    # check if current line contains meta information
    elif line[0] == esc_char:
        # check if this meta info line represents information about made
        # replacements
        if not memory.parse(line):
            # if it's other meta info -- simply print it
            foutput.fprint(line)
    # otherwise, line is a normal line and its umlauts should be restored with
    # corresponding update of replacement information
    else:
        line = misspelling_restorer.correct(line, memory)
        # print updated memory
        print_mem()
        # print replaced line
        foutput.fprint(line)

print_mem()
