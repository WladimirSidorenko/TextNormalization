#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module containing common sentment constants and methods.

@author: Wladimir Sidorenko <Uladzimir Sidarenka>

"""

##################################################################
# Imports
from alt_fio import AltFileInput
from ld.stringtools import COMMENT_RE
from conll import FEAT_NAME_SEP, FEAT_NAME_SEP_RE

import os
import re
import string


##################################################################
# Constants
ANY_TAG = "..."
SOURCE = "source"
TARGET = "target"
SENTIMENT = "sentiment"
EMOEXPRESSION = "emoexpression"
HASEMOEXPRESSION = "hasEmoExpression"
EMOKEY = (HASEMOEXPRESSION, HASEMOEXPRESSION)
EMOVALUE = {0: "1"}
FVALUE_SEP = '|'
PUNCT_RE = re.compile(r"^(?:" + '|'.join(
    [re.escape(c) for c in string.punctuation]) + ")+$")
DEFAULT_POLAR_LEXICA = [fn.format(**os.environ) for fn in
                        ["{SOCMEDIA_SEMDICT_DIR}/positive.zrch.smdict",
                         "{SOCMEDIA_SEMDICT_DIR}/negative.zrch.smdict"]]
DFLT_WORD2VEC = \
    os.path.join(os.environ.get("SOCMEDIA_LINGSRC"), "w2v.vectors")
DIG_RE = re.compile(r"^[\d.]*\d[\d.]*$")
SPACE_RE = re.compile("\s+")
F_V_SEP = ':'
F_V_SEP_RE = re.compile(F_V_SEP)
ESC_F_V_SEP = "__COLON__"


##################################################################
# Methods
def get_conll_mmax_features(iwords, conll_feat_list, mmax_feat_list):
    """For all words, obtain features from CONLL and those coming from MMAX.

    @param words - list of CONLL words
    @param conll_feat_list - target list for storing CONLL features
    @param mmax_feat_list - target list for storing MMAX features

    @return void

    """
    sentiment_start = -1
    sentiment_seen = False
    eexpression_seen = False
    for w_i, w in enumerate(iwords):
        # for each word create a dictionary of CONLL and MMAX features
        sentiment_seen = False
        conll_feats = {}
        conll_feat_list.append(conll_feats)
        mmax_feats = {}
        mmax_feat_list.append(mmax_feats)
        for fname, fvalue in w.pfeatures.iteritems():
            # check if feature comes from MMAX or not
            if FEAT_NAME_SEP_RE.search(fname):
                markname, markid, markattr = fname.split(FEAT_NAME_SEP)
                markname = markname.lower()
                markattr = markattr.lower()
                if markname == EMOEXPRESSION:
                    eexpression_seen = True
                    # if the `emo-expression` came after the sentiment span
                    # had begun, we add the feature 'hasEmoExpression=True'
                    # to all preceding sentiment words
                    if sentiment_seen:
                        for w_ii in xrange(sentiment_start, w_i + 1):
                            mmax_feat_list[w_ii][EMOKEY] = EMOVALUE
                elif markname == SENTIMENT:
                    if not sentiment_seen:
                        sentiment_seen = True
                        if sentiment_start < 0:
                            sentiment_start = w_i
                    if eexpression_seen:
                        mmax_feats[EMOKEY] = EMOVALUE
                ikey = (markname, markattr)
                # if we haven't seen such a markable before, create a
                # dictionary for it and put in this dictionary a mapping from
                # markable id to its position in the list of values
                if ikey not in mmax_feats:
                    mmax_feats[ikey] = {markid: fvalue}
                # otherwise, append a markable to an already created dict
                else:
                    # mmax_feats[ikey][0][markid] point to the element in list,
                    # which holds the attribute value for that markable id
                    mmax_feats[ikey][markid] = fvalue
            else:
                conll_feats[fname] = fvalue
        if not sentiment_seen:
            sentiment_start = -1
            eexpression_seen = False


def load_polar_dicts(dfnames):
    """Load polar words into polarity dictionary."""
    global polar_dict
    # iterate over name of polarity dictionaries
    finput = AltFileInput(*dfnames)
    word = tag = ""
    score = 0
    for iline in finput:
        if not COMMENT_RE.match(iline):
            word, tag, score = iline.split('\t')
            if tag == ANY_TAG:
                polar_dict[word.lower()] = abs(float(score))
            else:
                # abs(float(score))
                polar_dict[(word.lower(), tag)] = abs(float(score))


def wnormalize(a_word):
    """Normalize word's form.

    @param a_fname - name of file containing word embeddings
    @param a_remember_word - custom function for deciding whoch wotds to
      remember

    @return dict mapping words to their embeddings

    """
    return DIG_RE.sub('1', a_word.lower())
