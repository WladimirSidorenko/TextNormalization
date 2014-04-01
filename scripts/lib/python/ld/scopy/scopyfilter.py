#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
from util import get_index

##################################################################
class BaseFilter(object):
    def __init__(self, arg_finder):
        self._arg_finder = arg_finder
        self.next_on_success = None
        self.next_on_failure = None
        self._seen = set()

    def initialize(self, initialized=None):
        if initialized is None:
            initialized = set()
        elif self in initialized:
            return self
        initialized.add(self)
        if hasattr(self, 'on_initialize'):
            self.on_initialize()
        if self.next_on_success is not None:
            self.next_on_success.initialize(initialized=initialized)
        if self.next_on_failure is not None:
            self.next_on_failure.initialize(initialized=initialized)
        return self

    def on_success(self, filt):
        if filt is self:
            raise ValueError('cannot chain filter to itself')
        self.next_on_success = filt
        return self

    def on_failure(self, filt):
        if filt is self:
            raise ValueError('cannot chain filter to itself')
        self.next_on_failure = filt
        return self

    def next(self, filt):
        self.on_failure(filt)
        self.on_success(filt)
        return self

    def fire_success(self, sents, con):
        if self.next_on_success is not None:
            return self.next_on_success(sents, con)
        return sents

    def fire_failure(self, sents, con):
        if self.next_on_failure is not None:
            return self.next_on_failure(sents, con)
        return sents

class InOpaqueDirectSpeechFilter(BaseFilter):
    def __call__(self, sents, con):
        start = self._find_direct_speech(con)
        if start == -1:
            return self.fire_failure(sents, con)
        sents = [(i, sent) for (i, sent) in sents if i >= start]
        return self.fire_success(sents, con)

    def _find_direct_speech(self, con):
        layer = self._arg_finder._iter_paula_layer(
            'speechfinder.speechdcontent')
        for _, mark in layer:
            tokens = mark.ext
            start = get_index(tokens[0])
            if start > con.start_index:
                continue
            end = get_index(tokens[-1])
            if end < con.start_index or end < con.end_index:
                continue
            sent_idx = self._arg_finder._get_sent_index(tokens[0])
            print sent_idx, con.sent_index
            if sent_idx == con.sent_index:
                # Connective is in the first sentence of the segment.
                return -1
            return sent_idx
        return -1


class InOpaqueParenFilter(BaseFilter):
    PARENS = {
        '(': ')',
        '[': ']',
        '{': '}',
    }

    RIGHT_PARENS = frozenset(PARENS.values())

    MAX_LOOKBEHIND = 5

    def __call__(self, sents, con):
        span = self._in_paren(sents, con)
        if span is None:
            return self.fire_failure(sents, con)
        sents = [(i, sent) for (i, sent) in sents if span[0] <= i <= span[1]]
        return self.fire_success(sents, con)

    def _in_paren(self, sents, con):
        start, left_paren = self._find_left_paren(sents, con)
        if start == -1:
            return
        end = self._find_right_paren(con, left_paren)
        if end == -1 or start == end:
            # If start and end positions match, the connective is (trivially)
            # located in the first sentence of the parenthesized segment.
            return
        return start, end

    def _find_left_paren(self, sents, con):
        frontier = max(0, min(con.sent_index - len(sents),
                              con.sent_index - self.MAX_LOOKBEHIND))
        for i in xrange(con.sent_index - 1, frontier - 1, -1):
            sent = self._arg_finder._sentences[i].ext
            for j in xrange(len(sent) - 1, -1, -1):
                token = sent[j]
                strng = token.getString()
                if strng in self.RIGHT_PARENS:
                    return -1, None
                elif strng in self.PARENS:
                    # Make sure that all parts of a (possibly discontinuous)
                    # connective are included in the segment.
                    if get_index(token) > con.start_index:
                        return -1, None
                    return i, strng
        return -1, None

    def _find_right_paren(self, con, left_paren):
        right_paren = self.PARENS[left_paren]
        for i in xrange(con.sent_index, len(self._arg_finder._sents)):
            sent = self._arg_finder._sents[i]
            for token in sent.ext:
                strng = token.getString()
                if strng in self.PARENS:
                    return -1
                elif strng == right_paren:
                    # Make sure that all parts of a (possibly discontinuous)
                    # connective are included in the segment.
                    if get_index(token) < con.end_index:
                        return -1
                    return i
        return -1


class OutsideOpaqueZoneFilter(BaseFilter):
    PARENS = {
        '(': ')',
        '[': ']',
        '{': '}',
    }

    RIGHT_PARENS = frozenset(PARENS.values())

    def on_initialize(self):
        # 1. Find parenthesized sentences.
        self._paren_sents = set()
        sent_buffer = []
        right_paren = None
        embed_parens = []
        for sent_idx, sent in enumerate(self._arg_finder._sents):
            tokens = sent.ext
            if right_paren is None:
                strng = tokens[0].getString()
                if strng not in self.PARENS:
                    continue
                right_paren = self.PARENS[strng]
                sent_buffer.append(sent_idx)
                for index, token in enumerate(tokens[1:]):
                    if token.getString() == right_paren:
                        if index == len(tokens) - 1:
                            self._paren_sents.add(sent_idx)
                        sent_buffer = []
                        right_paren = None
                        break
            else:
                sent_buffer.append(sent_idx)
                for index, token in enumerate(sent.ext):
                    strng = token.getString()
                    if strng == right_paren:
                        if embed_parens and embed_parens[-1] == right_paren:
                            embed_parens.pop()
                        else:
                            self._paren_sents.update(sent_buffer)
                            sent_buffer = []
                            right_paren = None
                    elif strng in self.PARENS:
                        embed_parens.append(self.PARENS(strng))
        # 2. Find direct speech sentences.
        items = self._arg_finder._iter_paula_layer(
            'speechfinder.speechdcontent')
        self._paren_sents.update(self._arg_finder._get_sent_index(tok)
                                 for _, mark in items for tok in mark.ext)

    def __call__(self, sents, con):
        new_sents = [(i, sent)
                     for (i, sent) in sents if i not in self._paren_sents]
        if new_sents != sents:
            return self.fire_success(new_sents, con)
        return self.fire_failure(sents, con)


class ContrastFilter(BaseFilter):
    def on_initialize(self):
        self._frontiers = set(con.sent_index
                              for con in self._arg_finder._connectives if
                              self._is_frontier(con))

    def __call__(self, sents, con):
        sent_id = self._find_contrastive_connective(sents, con)
        if sent_id is None:
            return self.fire_failure(sents, con)
        sents = [(i, sent) for (i, sent) in sents if i >= sent_id]
        return self.fire_success(sents, con)

    def _find_contrastive_connective(self, sents, conn):
        for sent_idx, sent in sents:
            if sent_idx in self._frontiers:
                return sent_idx

    def _is_frontier(self, con):
        if not con.is_sent_initial:
            return False
        return con.string.lower() == 'aber' or 'contrast' in con.relations
