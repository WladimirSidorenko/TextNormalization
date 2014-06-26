#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Module providing methods and classes for correcting spelling mistakes.

Classes:
MisspellingRestorer() - class for doing correction of spelling
                        mistakes

"""

##################################################################
# Libraries
from alt_hunspell import Hunspell, DEFAULT_ENCD, DEFAULT_LANG
from alt_ngram import NGramProbDict, BOL, EOL
from ld.stringtools import adjust_case

import re
import sys

##################################################################
# Constants

##################################################################
# Methods
class MisspellingRestorer:
    """
    Constants:
    self.RULES  - a dict of correction rules.  Each key in this dictionary
        represents a regular expression which should capture a
        presumably misspelled word.  Values of this dictionary
        are 2-tuples in which the first element is a pointer to a
        checking function which should actually decide whether given
        word is a misspelling or not and the second element is a
        pointer to a correction function which should correct a
        misspelling.

    Instance Variables:
    self.dict - reference dictionary for lookup
    self.uniprob - reference to an object holding unigram statistics
    self.biprob - reference to an object holding bigram statistics

    Public Methods:
    __init__()  - initialize an instance of MisspellingRestorer
    correct() - perform misspelling correction and update information about
               offsets if needed

    """
    # private constants which represent regular expressions that should match
    # misspelled context
    #  elements of regular expressions
    __RE_ELEMENTS__ = dict(
        alpha   = ur'[A-züöä]',
        vowels  = ur'[aeuio]',
        lwbound = ur'(?:\s|\A)',
        rwbound = ur'(?:[-?.!,\s]|\Z)')

    # regular expressions representing suspicious contexts which might be a
    # spelling mistake
    __ARTICLE_RE__ = re.compile(ur"({lwbound}|irgend)'?(ne[mnrs]?{rwbound})".format(**__RE_ELEMENTS__), \
                                    re.I | re.L)
    __RAUS_RE__ = re.compile(ur"({lwbound})(rau(?:f|s\S)|rum(?!mel)|rüber|runter)", re.I | re.L)
    __STE_RE__ = re.compile(ur"({lwbound}{alpha}+st)(e)(?={rwbound})".format(**__RE_ELEMENTS__), \
                                re.I | re.L)
    __XN_RE__ = re.compile(ur"({lwbound}{alpha}+[^e\s])(n)(?={rwbound})".format(**__RE_ELEMENTS__), \
                               re.I | re.L)
    __XSSES_RE__ = re.compile(ur"({lwbound}{alpha}+s)s(es)(?={rwbound})".format(**__RE_ELEMENTS__), \
                                  re.I | re.L)
    __XSSER_RE__ = re.compile(ur"({lwbound}{alpha}+s)s(er)(?={rwbound})".format(**__RE_ELEMENTS__), \
                                  re.I | re.L)
    __XSTE_RE__ = re.compile(ur"({lwbound}{alpha}+st)(e)(?={rwbound})".format(**__RE_ELEMENTS__), \
                                 re.I | re.L)
    __XS_RE__ = re.compile(ur"({lwbound}{alpha}+[^e\W])(s)(?={rwbound})".format(**__RE_ELEMENTS__), \
                               re.I | re.L)
    __XES_RE__ = re.compile(ur"({lwbound}{alpha}+[^e\W])'?(s)(?={rwbound})".format(**__RE_ELEMENTS__), \
                                re.I | re.L)
    __XT_RE__ = re.compile(ur"({lwbound}{alpha}+[aeuio]t)(?={rwbound})".format(**__RE_ELEMENTS__), \
                               re.I | re.L)
    __XE_RE__ = re.compile(ur"({lwbound}{alpha}+[^\We])(?={rwbound})".format(**__RE_ELEMENTS__), \
                               re.I | re.L)

    # context of rules
    __MIDDLE_CAP_RE__ = re.compile(r"\s+[A-ZÖÜÄ]", re.L)
    __LEFT_WORD_RE__ = re.compile(r"(\S+)\s*\Z", re.I | re.L)
    __RIGHT_WORD_RE__ = re.compile(r"\A\s*(\S+)", re.I | re.L)

    def __init__(self, uniprob, biprob, encd = DEFAULT_ENCD, lang = DEFAULT_LANG):
        """Initialize an instance of MisspellingRestorer.

        Parameters:
        @param uniprob - an object with unigram probabilities
        @param biprob - an object holding bigram probabilities
        @param lang - language of Hunspell dictionary to be used
        @param encd - dictionary encoding
        """
        self.uniprob = uniprob
        self.biprob = biprob
        self.dict = Hunspell(encd, lang)
        # dictionary of rules
        self.RULES = {
            self.__XN_RE__:    (lambda mobj, mspan, *args: (not self.dict.check(mspan)) and \
                               self.dict.check(mobj.group(1) + 'e' + mobj.group(2)), \
                               lambda mobj: mobj.group(1) + 'e' + mobj.group(2)),
            self.__XSSES_RE__: (lambda mobj, mspan, *args: self.dict.check(mobj.group(1) + 't ' + \
                                                                          mobj.group(2)), \
                               lambda mobj: mobj.group(1) + 't ' + mobj.group(2)),
            self.__XSSER_RE__: (lambda mobj, mspan, *args: self.dict.check(mobj.group(1) + 't ' + \
                                                                          mobj.group(2)), \
                               lambda mobj: mobj.group(1) + 't ' + mobj.group(2)),
            self.__XSTE_RE__: (self.__xste_check__, lambda mobj: mobj.group(1) + " du"),
            self.__XS_RE__: (self.__xs_check__, lambda mobj: mobj.group(1) + ' e' + mobj.group(2)),
            self.__XES_RE__: (lambda mobj, mspan, *args: (not self.dict.check(mspan)) and \
                             self.dict.check(mobj.group(1) + 'e'), \
                             lambda mobj: mobj.group(1) + 'e e' + mobj.group(2)),
            self.__XE_RE__: (self.__xe_check__, lambda mobj: mobj.group(1) + 'e'),
            self.__XT_RE__: (lambda mobj, mspan, *args: \
                            (not self.dict.check(mspan)) and \
                            self.dict.check(mobj.group(1)[:-1] + 's'), \
                            lambda mobj: mobj.group(1)[:-1] + 's'),
            self.__ARTICLE_RE__: (lambda mobj, mspan, *args: True, \
                                 lambda mobj: mobj.group(1) + \
                                 adjust_case('ei' + mobj.group(2), mobj.group(2))),
            self.__RAUS_RE__: (lambda mobj, mspan, *args: True, \
                              lambda mobj: mobj.group(1) + adjust_case('he' + mobj.group(2), \
                                                                           mobj.group(2)))
        }
        self.WRONG_SEQS = sorted(self.RULES.keys())
        self.WRONG_SEQS_PTRN = [r.pattern for r in self.WRONG_SEQS]
        self.WRONG_SEQS_RE = re.compile("(?:" + '|'.join(self.WRONG_SEQS_PTRN) + ")", re.I | re.L)

    def correct(self, iline, memory):
        """Correct colloquial misspellings in `iline` and update offset information."""
        oline = ''
        start = 0
        mobj = self.WRONG_SEQS_RE.search(iline)
        mstart = mend = 0
        mspan = ''
        # if a suspicious sequence was met, start investigating it
        if mobj:
            while iline and mobj:
                # remember start and end position of the match
                mstart, mend = mobj.span()
                # append to the output line part of the input line before the
                # match
                oline = oline + iline[:mstart]
                # leave only part after the match in the original input line
                mspan = iline[mstart:mend]
                # truncate line to the part after the match
                iline = iline[mend:]
                # check if any rule applies to the captured span and return the
                # result of transformation if any rule does so
                checkstatus, result = self.__check_rule__(mspan, oline, iline)
                # if the span was modified
                if checkstatus:
                    # add modified span to the output
                    oline += result
                    # update memory information by the difference in length of
                    # matched span and its replacement
                    memory.update(len(oline), len(result) - (mend - mstart))
                else:
                    # otherwise append matched span to the output line
                    oline += mspan
                # check again if any other rule could be applied to the remaining
                # part of the line
                start += mend
                mobj  = self.WRONG_SEQS_RE.search(iline)
        oline += iline
        return oline

    def __check_rule__(self, istring, leftcontext, rightcontext):
        """Check if an RE from keys in RULES produced a match and return modified string."""
        leftword = rightword = ''
        # iterate over all possible regular expressions in RULES
        for kre in self.WRONG_SEQS:
            # check if any of them matches and associated `checkfunc' produces a
            # True value
            mobj = kre.match(istring)
            if mobj and self.RULES[kre][0](mobj, istring, leftcontext, rightcontext):
                # return result on success
                return True, self.RULES[kre][1](mobj)
        # otherwise return None
        return False, ''

    # additional methods for checking rule context
    def __xe_check__(self, mobj, mspan, leftcontext = '', rightcontext = ''):
        """Check if input data captured by __XE_RE__ actually got a misspelling."""
        # this is only done for better readability
        orig = mspan
        newform = mspan + 'e'
        # check linguistic conditions
        ret = not self.dict.check(orig) and self.dict.check(newform)
        # print >> sys.stderr, "ret (ling) is", ret
        # check probabilistic conditions
        if ret:
            # left and right words for N-grams
            lw = self.__LEFT_WORD_RE__.search(leftcontext)
            lw = lw.group(1) if lw else BOL
            rw = self.__RIGHT_WORD_RE__.match(rightcontext)
            rw = rw.group(1) if rw else EOL

            # calculate probabilities of n-grams for replacement and for original
            # form
            prob1 = sum([self.biprob.get_prob(lw, newform), \
                             self.uniprob.get_prob(newform), \
                             self.biprob.get_prob(newform, rw)])
            prob2 = sum([self.biprob.get_prob(lw, orig), \
                             self.uniprob.get_prob(orig), \
                             self.biprob.get_prob(orig, rw)])
            ret =  prob1 >= prob2
        # return the result
        return ret

    def __xs_check__(self, mobj, mspan, leftcontext = '', rightcontext = ''):
        """Check if input data captured by __XS_RE__ actually got a misspelling."""
        # this variables are only introduced for better readability
        orig = mspan
        newform = mobj.group(1)
        # check linguistic conditions
        ret = (not self.dict.check(mspan)) and self.dict.check(mobj.group(1))
        # check probabilistic conditions
        if ret:
            # left and right words for N-grams
            lw = self.__LEFT_WORD_RE__.search(leftcontext)
            lw = lw.group(1) if lw else BOL
            rw = self.__RIGHT_WORD_RE__.match(rightcontext)
            rw = rw.group(1) if rw else EOL
            # calculate probabilities of n-grams for replacement and for original
            # form
            prob1 = sum([self.biprob.get_prob(lw, orig), \
                             self.uniprob.get_prob(orig), \
                             self.biprob.get_prob(orig, rw)])
            prob2 = sum([self.biprob.get_prob(lw, newform), \
                             self.uniprob.get_prob(newform), \
                             self.biprob.get_prob(newform, rw)])
            ret =  prob1 < prob2
        return ret

    def __xste_check__(self, mobj, mspan, leftcontext = '', rightcontext = ''):
        """Check if input data captured by __XSTE_RE__ actually got a misspelling."""

        # left and right words for N-grams
        lw = self.__LEFT_WORD_RE__.search(leftcontext)
        lw = lw.group(1) if lw else BOL
        rw = self.__RIGHT_WORD_RE__.match(rightcontext)
        rw = rw.group(1) if rw else EOL

        orig    = mspan
        newform = mobj.group(1)

        prob1 = sum([self.biprob.get_prob(lw, orig), \
                         self.uniprob.get_prob(orig), \
                         self.biprob.get_prob(orig, rw)])
        prob2 = sum([self.biprob.get_prob(lw, newform), \
                         self.uniprob.get_prob(newform), \
                         self.biprob.get_prob(newform, "du")])
        return self.dict.check(mobj.group(1)) and \
            ((not mspan) or prob1 < prob2)
