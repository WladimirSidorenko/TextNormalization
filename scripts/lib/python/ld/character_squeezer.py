#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Module providing methods and classes for correcting multiply repeated
characters.

Constants:
DEFAULT_LENGTH_PROB - default path to file with statistics about
                      elongated words

Classes:
CharacterSqueezer() - class for correcting multiply repeated characters

"""

##################################################################
# Libraries
from alt_hunspell import Hunspell, DEFAULT_ENCD, DEFAULT_LANG
from alt_ngram import NGramProbDict
from stringtools import adjust_case
from . import repeated_chars, UNIGRAM_DEFAULT_FILE, BIGRAM_DEFAULT_FILE
from tokenizer import EOS_TAG

import os
import pickle
import re
import sys

##################################################################
# Constants
MAX_CANDIDATES = -1             # maximum number of suggestion candidates
DEFAULT_LENGTH_PROB = "{SOCMEDIA_LINGBIN}/lengthened_stat.pckl".format(**os.environ)
candidates = set([])

##################################################################
# Class
class CharacterSqueezer:
    """
    Constants:

    Instance Variables:
    self.dict - an instance of reference dictionary for checking words
    self.probs - a dictionary with probabilities of normalized forms of
                 elongated words
    self.uniprob - a dictionary with unigram probabilities
    self.biprob - a dictionary with bigram probabilities

    Public Methods:
    __init__()  - initialize an instance of CharacterSqueezer
    has_repetition() - test function returning true if a word contains
                       repated characters
    squeeze_characters() - normalize word with elongated characters

    """

    def __init__(self, single = True, probs = DEFAULT_LENGTH_PROB, \
                     uprob = UNIGRAM_DEFAULT_FILE, biprob = BIGRAM_DEFAULT_FILE, \
                     diclang = DEFAULT_LANG, dicenc = DEFAULT_ENCD):
        """Initialize an instance of CharacterSqueezer.

        @param single - boolean flag indicating whether only one suggestion should
                        be generated
        @param diclang - language of reference dictionary
        @param dicenc - encoding of reference dictionary
        @param probs - name of a pickle file with probabilities of normalized
                       forms of elongated words
        @param uprob - a pickle file with unigram probabilities dictionary
        @param biprob - a pickle file with bigram probabilities dictionary

        """
        self.single = single
        self.probs = self.__read_file__(probs)
        self.uniprob = self.__read_file__(uprob)
        self.biprob = self.__read_file__(biprob)
        self.dict = Hunspell(dicenc, diclang)

    def has_repetition(self, iword):
        """Return bool indicating if any squeezing should be performed at all.

        @param iword - input word which should be analyzed

        @return boolean indicating whether word has elongated characters

        """
        # TODO: check if sequence of repeated characters is located on the
        # boundary of a compound
        return repeated_chars.THREE_LETTERS_RE.search(iword) and \
            not repeated_chars.LEGAL_REPETITION_RE.match(iword)

    def squeeze_characters(self, iword, leftword, rightword):
        """Generate most probable squeezed candidate for iword and return it.

        @param iword - input word with elongations whose characters should be
                       squeezed
        @param leftword - word appearing to the left of iword in sentence
        @param rightword - word appearing to the right of iword in sentence

        @return modified input word
        """
        candidates = self.__prune__(iword, self.__generate_candidates__(iword), \
                                        leftword, rightword)
        # check if any candidates were generated
        if not candidates:
            return iword
        # if requested or if only one candidate was generated, output
        # only the first candidate for replacement
        elif self.single or len(candidates) == 1:
            return adjust_case(candidates[0][0], iword)
        else:
            stat = u""
            # convert candidates and their probs to strings
            for candidate, prob in candidates:
                stat += u"\t" + adjust_case(candidate, iword) + u" " + unicode(prob)
            return iword + stat

    def __generate_candidates__(self, iword):
        """Generate normalization candidates by squeezing repeating letters."""
        # squeeze occurrences of same letters which repeat more than 3
        # times in sequence to just 3 repeating occurrences
        iword = repeated_chars.GT_THREE_LETTERS_RE.sub(r"\1\2", iword)
        # generate all possible candidates
        return sorted(self.__generate_candidates_helper__(iword))[:MAX_CANDIDATES]

    def __generate_candidates_helper__(self, iword, pos = 0):
        """Look for all occurrences of repeated letters and squeeze them."""
        ret   = []
        m_obj = repeated_chars.REPEATED_LETTERS_RE.search(iword[pos:])
        if m_obj:
            start = pos + m_obj.start()
            end   = pos + m_obj.end()
            # iterate on original line with increased pos
            ret += self.__generate_candidates_helper__(iword, end)
            # change line and iterate on changed version
            iword = iword[:end - 1] + iword[end:]
            ret += self.__generate_candidates_helper__(iword, start)
        else:
            ret.append(iword)
        return ret

    def __equiprobable__(self, icandidates):
        """Assign equal probabilities to all elements of icandidates."""
        # divide 1 by number of candidates
        prob = 1.0 / len(icandidates)
        # assign this probability to all the candidates
        return [(candidate, prob) for candidate in icandidates]

    def __assign_probs__(self, icandidates, leftword, rightword):
        """Assign n-gram probabilities to all elements of icandidates."""
        # divide 1 by number of candidates
        prob = 1.0 / len(icandidates)
        # assign this probability to all the candidates
        icandidates = [(candidate, sum([self.biprob.get_prob(leftword, candidate), \
                                            self.uniprob.get_prob(candidate), \
                                            self.biprob.get_prob(candidate, rightword),])) \
                           for candidate in icandidates]
        icandidates.sort(key=lambda x: x[1], reverse = True)
        return icandidates

    def __prune__(self, orig_word, candidates, leftword, rightword):
        """Filter-out unlikely candidates based on heuristics."""
        # method squeeze() is defined in ld.repeated_chars
        stat_key = repeated_chars.squeeze(orig_word.lower())
        # lookup generated candidates in dictionary
        dictforms = [form for form in candidates if self.dict.check(form)]
        # if any of the candidates were found in dictionary, return only
        # found candidates
        if dictforms:
            # assign equal probability to all forms found in dictionary
            # (subject to change)
            # return equiprobable(dictforms)
            return self.__assign_probs__(dictforms, leftword, rightword)
        # otherwise, check if we have gathered some statistics for given
        # squeezed form and rely solely on statistics if yes
        # as a default solution, make original word the 1-st candidate in list,
        # then assign equal probabilities to all the list elements and return list.
        elif stat_key in self.probs:
            # if we've seen the value on corpus, return mappings from
            # corpus along with their probabilities. (If no stat_file was
            # specified as argument prob_table will be an empty dict)
            return self.probs[stat_key]
        else:
            # make original word the first candidate in the list of candidates
            if orig_word in candidates:
                orig_word_idx = candidates.index(orig_word)
                # if we found orig word in our candidate list, swap it with the
                # very 1-st list element
                candidates[0], candidates[orig_word_idx] = \
                    candidates[orig_word_idx], candidates[0]
            else:
                candidates[0] = orig_word
        return self.__equiprobable__(candidates)

    def __read_file__(self, fname):
        """Read and return pickle object from file named `fname`.

        @param fname - name of the file containing pickle object

        @return pickle object stored in the file
        """
        istream =  open(fname, 'r')
        p_obj = pickle.load(istream)
        istream.close()
        return p_obj
