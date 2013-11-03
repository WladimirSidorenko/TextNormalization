#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Script for correcting colloquial writings.

"""

##################################################################
# Libraries
import os
import re
import sys
import pickle

from alt_argparse import argparser
from alt_fio import AltFileInput, AltFileOutput
from alt_hunspell import Hunspell
from alt_ngram import NGramProbDict, BOL, EOL
from ld import p2p
from ld.stringtools import adjust_case
from replacements import Memory
from tokenizer import EMSG_TAG

##################################################################
# Constants
# regular expressions capturing wrong contexts
re_elements = dict(
    alpha   = ur'[A-züöä]',
    vowels  = ur'[aeuio]',
    lwbound = ur'(?:\s|\A)',
    rwbound = ur'(?:[-?.!,\s]|\Z)'
)

ARTICLE_RE = re.compile(ur"({lwbound}|irgend)'?(ne[mnrs]?{rwbound})".format(**re_elements), \
                            re.I | re.L)
RAUS_RE    = re.compile(ur"({lwbound})(rau(?:f|s\S)|rum(?!mel)|rüber|runter)", re.I | re.L)
STE_RE     = re.compile(ur"({lwbound}{alpha}+st)(e)(?={rwbound})".format(**re_elements), \
                            re.I | re.L)
XN_RE      = re.compile(ur"({lwbound}{alpha}+[^e\s])(n)(?={rwbound})".format(**re_elements), \
                            re.I | re.L)
XSSES_RE   = re.compile(ur"({lwbound}{alpha}+s)s(es)(?={rwbound})".format(**re_elements), \
                            re.I | re.L)
XSSER_RE   = re.compile(ur"({lwbound}{alpha}+s)s(er)(?={rwbound})".format(**re_elements), \
                            re.I | re.L)
XSTE_RE    = re.compile(ur"({lwbound}{alpha}+st)(e)(?={rwbound})".format(**re_elements), \
                            re.I | re.L)
XS_RE      = re.compile(ur"({lwbound}{alpha}+[^e\W])(s)(?={rwbound})".format(**re_elements), \
                            re.I | re.L)
XES_RE     = re.compile(ur"({lwbound}{alpha}+[^e\W])'?(s)(?={rwbound})".format(**re_elements), \
                            re.I | re.L)
XT_RE      = re.compile(ur"({lwbound}{alpha}+[aeuio]t)(?={rwbound})".format(**re_elements), \
                            re.I | re.L)
XE_RE      = re.compile(ur"({lwbound}{alpha}+[^\We])(?={rwbound})".format(**re_elements), \
                            re.I | re.L)

MIDDLE_CAP_RE = re.compile(r"\s+[A-ZÖÜÄ]", re.L)
LEFT_WORD  = re.compile(r"(\S+)\s*\Z", re.I | re.L)
RIGHT_WORD = re.compile(r"\A\s*(\S+)", re.I | re.L)

# dictionary checker
DictChecker = Hunspell()

# default n-gram files
UNIGRAM_DEFAULT_FILE = "{SOCMEDIA_LINGBIN}/unigram_stat.pckl".format(**os.environ)
BIGRAM_DEFAULT_FILE  = "{SOCMEDIA_LINGBIN}/bigram_stat.pckl".format(**os.environ)

##################################################################
# Processing Arguments
# note: some options are already set up by alt_argparser
argparser.description="""Utility for correcting colloquial spellings in text"""
argparser.add_file_argument("-b", "--bigram-prob-file", help="file with bigram probabilities", \
                                default = BIGRAM_DEFAULT_FILE)
argparser.add_file_argument("-u", "--unigram-prob-file", help="file with unigram probabilities", \
                                default = UNIGRAM_DEFAULT_FILE)
argparser.add_argument("-v", "--verbose", help="switch verbose statistics mode on", \
                           action = "store_true")
argparser.add_argument("-x", "--skip-xml", help="skip XML tags", \
                           action = "store_true")
args = argparser.parse_args()

unigram_prob = pickle.load(args.unigram_prob_file)
args.unigram_prob_file.close()

bigram_prob  = pickle.load(args.bigram_prob_file)
args.bigram_prob_file.close()

##################################################################
# Methods
def xe_check(mobj, mspan, leftcontext = '', rightcontext = ''):
    """Check if input data satisfies to conditions to be applied."""
    # this is only done for better readability
    orig = mspan
    newform = mspan + 'e'

    # print >> sys.stderr, "orig is", orig
    # print >> sys.stderr, "newform is", newform
    # check linguistic conditions
    ret = not DictChecker.check(orig) and DictChecker.check(newform)
    # print >> sys.stderr, "ret (ling) is", ret
    # check probabilistic conditions
    if ret:
        # left and right words for N-grams
        lw = LEFT_WORD.search(leftcontext)
        lw = lw.group(1) if lw else BOL

        rw = RIGHT_WORD.match(rightcontext)
        rw = rw.group(1) if rw else EOL

        # calculate probabilities of n-grams for replacement and for original
        # form
        prob1 = sum([bigram_prob.get_prob(lw, newform), \
                         unigram_prob.get_prob(newform), \
                         bigram_prob.get_prob(newform, rw)])
        prob2 = sum([bigram_prob.get_prob(lw, orig), \
                         unigram_prob.get_prob(orig), \
                         bigram_prob.get_prob(orig, rw)])
        ret =  prob1 >= prob2
        # print >> sys.stderr, "ret (prob) is", ret
        # print >> sys.stderr, prob1
        # print >> sys.stderr, prob2
    # return the result
    return ret

def xs_check(mobj, mspan, leftcontext = '', rightcontext = ''):
    """Check if input data satisfies conditions to be applied."""
    # this is only done for better readability
    orig = mspan
    newform = mobj.group(1)

    # print >> sys.stderr, "orig is", orig
    # print >> sys.stderr, "newform is", newform
    # check linguistic conditions
    ret = (not DictChecker.check(mspan)) and DictChecker.check(mobj.group(1))
    # print >> sys.stderr, "ret (ling) is", ret
    # check probabilistic conditions
    if ret:
        # left and right words for N-grams
        lw = LEFT_WORD.search(leftcontext)
        lw = lw.group(1) if lw else BOL

        rw = RIGHT_WORD.match(rightcontext)
        rw = rw.group(1) if rw else EOL

        # calculate probabilities of n-grams for replacement and for original
        # form
        prob1 = sum([bigram_prob.get_prob(lw, orig), \
                         unigram_prob.get_prob(orig), \
                         bigram_prob.get_prob(orig, rw)])
        prob2 = sum([bigram_prob.get_prob(lw, newform), \
                         unigram_prob.get_prob(newform), \
                         bigram_prob.get_prob(newform, rw)])
        ret =  prob1 < prob2
        # print >> sys.stderr, "ret (prob) is", ret
    # return the result
    return ret

def xste_check(mobj, mspan, leftcontext = '', rightcontext = ''):
    """Check if input data satisfies conditions to be applied."""
    # left and right words for N-grams
    lw = LEFT_WORD.search(leftcontext)
    lw = lw.group(1) if lw else BOL

    rw = RIGHT_WORD.match(rightcontext)
    rw = rw.group(1) if rw else EOL

    orig    = mspan
    newform = mobj.group(1)

    prob1 = sum([bigram_prob.get_prob(lw, orig), \
                     unigram_prob.get_prob(orig), \
                     bigram_prob.get_prob(orig, rw)])
    prob2 = sum([bigram_prob.get_prob(lw, newform), \
                     unigram_prob.get_prob(newform), \
                     bigram_prob.get_prob(newform, "du")])
    return DictChecker.check(mobj.group(1)) and \
        ((not mspan) or prob1 < prob2)

def capital_in_middle(istring):
    """Check whether istring starts with a space and has a capital letter at
    the beginning."""
    return MIDDLE_CAP_RE.match(istring)

##################################################################
# Rewriting rules
RULES = {
    XN_RE:    (lambda mobj, mspan, *args: (not DictChecker.check(mspan)) and \
                   DictChecker.check(mobj.group(1) + 'e' + mobj.group(2)), \
                   lambda mobj: mobj.group(1) + 'e' + mobj.group(2)),
    XSSES_RE: (lambda mobj, mspan, *args: DictChecker.check(mobj.group(1) + 't ' + mobj.group(2)), \
                   lambda mobj: mobj.group(1) + 't ' + mobj.group(2)),
    XSSER_RE: (lambda mobj, mspan, *args: DictChecker.check(mobj.group(1) + 't ' + mobj.group(2)), \
                   lambda mobj: mobj.group(1) + 't ' + mobj.group(2)),
    XSTE_RE: (xste_check, lambda mobj: mobj.group(1) + " du"),
    XS_RE: (xs_check, lambda mobj: mobj.group(1) + ' e' + mobj.group(2)),
    XES_RE: (lambda mobj, mspan, *args: (not DictChecker.check(mspan)) and \
                DictChecker.check(mobj.group(1) + 'e'), \
                lambda mobj: mobj.group(1) + 'e e' + mobj.group(2)),
    XE_RE: (xe_check, lambda mobj: mobj.group(1) + 'e'),
    XT_RE: (lambda mobj, mspan, *args: \
                (not DictChecker.check(mspan)) and DictChecker.check(mobj.group(1)[:-1] + 's'), \
                lambda mobj: mobj.group(1)[:-1] + 's'),
    ARTICLE_RE: (lambda mobj, mspan, *args: True, \
                     lambda mobj: mobj.group(1) + \
                     adjust_case('ei' + mobj.group(2), mobj.group(2))),
    RAUS_RE: (lambda mobj, mspan, *args: True, \
                  lambda mobj: mobj.group(1) + adjust_case('he' + mobj.group(2), \
                                                               mobj.group(2)))
}

WRONG_SEQS = sorted(RULES.keys())
WRONG_SEQS_PTRN = [r.pattern for r in WRONG_SEQS]
WRONG_SEQS_RE   = re.compile("(?:" + '|'.join(WRONG_SEQS_PTRN) + ")", re.I | re.L)

def apply_rule(istring, leftcontext, rightcontext):
    """Check which RE from keys in RULES produced a match and return it."""
    leftword = rightword = ''
    # iterate over all possible regular expressions in RULES
    for kre in WRONG_SEQS:
        # check if any of them matches and associated `checkfunc' produces a
        # True value
        mobj = kre.match(istring)
        if mobj and RULES[kre][0](mobj, istring, leftcontext, \
                                      rightcontext):
            # return result on success
            return True, RULES[kre][1](mobj)
    # otherwise return None
    return False, ''

def custom_print(iline):
    """Print iline if it is not empty."""
    if not isinstance(iline, basestring):
        iline = unicode(iline)
    if iline:
        foutput.fprint(iline)

##################################################################
# Main
foutput   = AltFileOutput(encoding = args.encoding, \
                              flush = args.flush)
finput    = AltFileInput(*args.files, \
                              print_func = foutput.fprint, \
                              errors = 'replace')
oline  = ''
mobj   = None
mspan  = ''
start = mstart = mend = 0
re_key = None
checkstatus = False
replfunc    = None

msg_lines_before_mem = []
msg_lines_after_mem  = []
msg_memory   = Memory()
memline_seen = False
skip_line    = args.skip_line

for line in finput:
    if line == skip_line or line == EMSG_TAG:
        # print and forget everything we have seen before mem line
        custom_print('\n'.join(msg_lines_before_mem))
        del msg_lines_before_mem[:]
        # print and forget memory
        custom_print(msg_memory)
        msg_memory.forget_all()
        # print and forget everything we have seen after mem line
        custom_print('\n'.join(msg_lines_after_mem))
        del msg_lines_after_mem[:]
        # print current line
        foutput.fprint(line)
        # reset variable
        memline_seen = False
    # check if current line can be parsed by Memory()
    elif not msg_memory.parse(line):
        start = 0
        mobj = WRONG_SEQS_RE.search(line)
        if mobj:
            oline = ''
            while line and mobj:
                mstart, mend = mobj.span()
                # append to the output line part of the input line before the
                # match
                oline = oline + line[:mstart]
                # leave only part after the match in the original input line
                mspan = line[mstart:mend]
                line = line[mend:]
                # check if any rule applies and return the result of
                # transformation if any rule does so
                checkstatus, result = apply_rule(mspan, oline, line)
                # check if transformation should be applied
                if checkstatus:
                    oline += result
                    # update memory
                    # print >> sys.stderr, "Updating memory:", start, result, mend, mstart
                    msg_memory.update(start, len(result) - (mend - mstart))
                else:
                    # otherwise append matched span to the output line
                    oline += mspan
                # check again if any other rule could be applied to the remaining
                # part of the line
                start += mend
                mobj  = WRONG_SEQS_RE.search(line)
            line = oline + line
        if memline_seen:
            msg_lines_after_mem.append(line)
        else:
            msg_lines_before_mem.append(line)
    else:
        # if line could be parsed by memory, remember that we have seen it
        memline_seen = True

# print the rest if any
custom_print('\n'.join(msg_lines_before_mem))
custom_print(msg_memory)
custom_print('\n'.join(msg_lines_after_mem))