#!/usr/bin/env python2.7

"""
BEWARE: Thuis is a quick and dirty implementation of n-gram probabilities. It
was only superficially tested, not thoroughly optimized and is currently using
suboptimal add-lambda smoothing whereas modified Kneser-Ney should be
preferred.
"""

##################################################################
# Libraries
import re
import string
import sys

from collections import Counter, defaultdict
from math import log

##################################################################
# Constants
BOL = "<bol>"
EOL = "<eol>"

PUNCT_RE = re.compile(r"(?:" + '|'.join([re.escape(c) for c in string.punctuation]) + ")")
SPACE_RE = re.compile(r"\s\s+")

LAMBDA = 0.05

def __adjust_key__(key):
    """Perform necessary operations on key string to match the stored form."""
    key = SPACE_RE.sub(" ", PUNCT_RE.sub(" ", key).strip()).lower()
    return key

##################################################################
# Class
class NGramStatDict(defaultdict):
    """Class holding statistics on n-grams."""

    def __init__(self, n = 1):
        "Initialize an NGramProb for holding n-grams of size N."
        self.N = n
        self.max_width = 0.0
        if self.N == 1:
            self.default_factory = lambda: 0
            self.incr = self.__incr_one__
            self.get_max_width = self.__get_max_width_one__
        elif self.N > 1:
            self.default_factory = lambda: NGramStatDict(self.N - 1)
            self.incr = self.__incr_X__
            self.get_max_width = self.__get_max_width_X__
        else:
            raise Exception("Invalid length of N-gram specified: {} (should be at least 1)".format(n))

    def __incr_one__(self, key = []):
        """Increase counter for key by one."""
        key[0] = __adjust_key__(key[0])
        self[key[0]] += 1

    def __incr_X__(self, key = []):
        """Pass incr() to child element."""
        key[0] = __adjust_key__(key[0])
        self[key[0]].incr(key[1:])

    def __get_max_width_one__(self):
        """Compute maximal number of keys in a unigram."""
        self.max_width = float(len(self.keys()))
        self.get_max_width = lambda: self.max_width
        return self.max_width

    def __get_max_width_X__(self):
        """Compute maximal number of keys in an N-gram (N > 1)."""
        self.max_width = max([v.get_max_width() for k,v in self.iteritems()])
        self.get_max_width = lambda: self.max_width
        return self.max_width


class NGramProbDict(defaultdict):
    """Class for storing N-gram probabilities."""

    """TODO: Modified Kneser-Ney should be used on top of add-lambda."""

    def __init__(self, ngram_stat = None, norm_factor = 0.0):
        "Initialize probability table from statistics dictionary."
        # to allow restoration from serialization in which case __init__() with
        # no arguments is invoked
        if not ngram_stat:
            pass
        elif ngram_stat.N >= 1:
            # call parent's constructor
            super(NGramProbDict, self).__init__()
            self.N = ngram_stat.N
            if self.N > 1:
                self.total = 0.0
                for k,v in ngram_stat.iteritems():
                    self[k] = NGramProbDict(v, norm_factor)
                    self.total += self[k].total
                    self.get_prob = self.__get_prob_X__
            else:
                self.total = sum(ngram_stat.values()) + LAMBDA * norm_factor
                self.get_prob = self.__get_prob_one__
                for k,v in ngram_stat.iteritems():
                    # calculate log probabilities instead of raw probabilities
                    self[k] = log((v + LAMBDA) / self.total)
            # calculate a default log probability for elements which are
            # missing
            self.default = log(LAMBDA / self.total)
        else:
            raise Exception("Invalid size of N-gram.")

    def __get_prob_one__(self, *key):
        """Return default probability on key."""
        ikey = __adjust_key__(key[0])
        if ikey in self:
            return self[ikey]
        else:
            return self.default

    def __get_prob_X__(self, *key):
        """Return default probability on key."""
        ikey = __adjust_key__(key[0])
        if ikey in self:
            return self[ikey].get_prob(*key[1:])
        else:
            return self.default

    def __missing__(self, key):
        """Return default value if key is missing."""
        ikey = __adjust_key__(key)
        if ikey in self:
            return self[ikey]
        else:
            return self.default

    def __getstate__(self):
        """Specialize how this object should be serialized."""
        res = self.__dict__.copy()
        del res["get_prob"]
        return res

    def __setstate__(self, idict):
        """Specialize how this object should be restored from serialization."""
        self.__dict__ = idict
        self.get_prob = self.__get_prob_one__ if self.N == 1 else \
            self.__get_prob_X__

    def __reduce__(self):
        """Specialize how this object should be reconstructed upon
        serialization."""
        return (NGramProbDict, (), self.__getstate__(), None, self.iteritems())
