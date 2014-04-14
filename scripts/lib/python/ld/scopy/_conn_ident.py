#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for finding connectives in CONLL sentences.

This module provides the following classes:

ConnectiveFinder() - class implementing search procedure for finding
                     connectives

"""

import sys

class ConnectiveFinder(object):
    def __init__(self, lex):
        """Load list of connectors from extrnal XML file."""
        self._connectives = []
        for orth in lex.xpath('entry/orth'):
            self._connectives.append([part.text.strip() for part in orth])

    def find(self, a_tokens):
        """Find connectors in a list of tokens.

        Iterate over all tokens in a list, find connectors, and return a set of
        indices of tokens which represent those connectors.

        """
        # temporary array for storing indices of connector tokens
        ctoks = []
        # temporary set for storing connectors which are already known
        known_ctoks = set()
        connector_tokens = []
        # iterate over all tokens
        # print >> sys.stderr, "a_tokens = ", repr(a_tokens)
        for i in xrange(len(a_tokens)):
            # skip tokens which were already recognized as parts of multiword
            # connectives
            # print >> sys.stderr, "Considering token:", self._get_tkn_wrd(a_tokens, i)
            # print >> sys.stderr, "known_ctoks = ", repr(known_ctoks)
            # print >> sys.stderr, "a_tokens[i][0] =", repr(a_tokens[i][0])
            if a_tokens[i][0] in known_ctoks:
                # print >> sys.stderr, "Skipping token..."
                continue
            # get a list of token id's that represent connectives
            self._find_connectives(ctoks, a_tokens, i)
            # append the list of connector tokens to the return value
            if ctoks:
                connector_tokens.extend(ctoks)
                # remember the list of connector tokens for further checking
                known_ctoks.update(set([tok_id for toklst in ctoks \
                                            for tok_id, tok in toklst]))
                del ctoks[:]
        known_ctoks.clear()
        return connector_tokens

    def _find_connectives(self, a_res_tokens, a_itokens, a_i):
        """Try to find connective at position a_i in token list a_itokens."""
        found = True
        tokens = []
        pos = a_i
        # print >> sys.stderr, "a_i is", a_i
        # iterate over all known connectives
        for connective in self._connectives:
            found = True
            # iterate over each part of a connective (because connectives can
            # consist of multiple parts)
            for part in connective:
                # a_tokens, a_i, a_tok_id, a_tok
                # print >> sys.stderr, repr(part)
                pos, matched_tokens = self._match_part(part, a_itokens, pos)
                if pos == -1:
                    # print >> sys.stderr, "Part not matched"
                    found = False
                    break
                # print >> sys.stderr, "Part matched"
                tokens.extend(matched_tokens)
            if found:
                a_res_tokens.append(tokens)
                break
            else:
                # reset search position
                pos = a_i

    def _match_part(self, a_part, a_itokens, a_pos):
        """Find all parts of a multiword connective."""
        tokens = []
        j = 1
        # print >> sys.stderr, "a_part is", repr(a_part)
        markers = a_part.split()
        form = ""
        # iterate over succeeding tokens and try to match them against parts of
        # a connector
        try:
            if self._get_tkn_wrd(a_itokens, a_pos) == markers[0]:
                tokens.append(a_itokens[a_pos])
                found = True
                a_pos += 1
                # print >> sys.stderr, "markers is", repr(markers)
                for marker in markers[1:]:
                    try:
                        form = self._get_tkn_wrd(a_itokens, a_pos)
                    except IndexError:
                        return -1, None
                    if form != marker:
                        found = False
                        break
                    tokens.append(a_itokens[a_pos])
                    a_pos += 1
                if found:
                    return a_pos, tokens
        except IndexError:
            return -1, None
        return -1, None

    def _get_tkn_id(self, a_itokens, a_pos):
        """Return id token located at given index."""
        return a_itokens[a_pos][0]

    def _get_tkn_wrd(self, a_itokens, a_pos):
        """Return word form of token located at given index."""
        return a_itokens[a_pos][-1][0].form
