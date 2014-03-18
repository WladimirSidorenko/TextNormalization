#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import codecs
from collections import defaultdict
from functools import partial


def match(tokens, *search, **options):
    if options.get('reverse'):
        tokens = reversed(tokens)
        search = reversed(search)
    attr = options.get('attr', 'form')
    for token, item in zip(tokens, search):
        if isinstance(item, basestring):
            if token[attr] != item:
                return False
        elif not any(elem == token[attr] for elem in item):
            return False
    return True


class Trie(object):
    class NotFinal(Exception):
        pass

    def __init__(self):
        self.start = 0
        self._trans = defaultdict(dict)
        self._final = {}
        self._last_state = self.start

    def add_word(self, word, output=None):
        state = self.start
        for char in word:
            state = self.add_trans(state, char)
        self.set_final(state, output)
        return state

    def __contains__(self, string):
        state = self.start
        for char in string:
            next_state = self.get_trans(state, char)
            if next_state is None:
                return
            state = next_state
        return state in self._final

    _SENTINEL = object()

    def get(self, string, holistic=True, default=_SENTINEL):
        last_output = self._SENTINEL
        state = self.start
        for char in string:
            if not holistic and state in self._final:
                last_output = self.get_olabel(state)
            next_state = self.get_trans(state, char)
            if next_state is None:
                if last_output is not self._SENTINEL:
                    return last_output
                elif default is not self._SENTINEL:
                    return default
                raise self.NotFinal
            state = next_state
        try:
            return self.get_olabel(state)
        except self.NotFinal:
            if default is not self._SENTINEL:
                return default
            raise

    def get_state(self):
        self._last_state += 1
        return self._last_state

    def set_final(self, state, olabel=None):
        self._final[state] = olabel
        return self

    def is_final(self, state):
        return state in self._final

    def get_olabel(self, state):
        try:
            return self._final[state]
        except KeyError:
            raise self.NotFinal(state)

    def set_olabel(self, state, olabel):
        if state not in self._final:
            raise self.NotFinal(state)
        self._final[state] = olabel

    def add_trans(self, state, ilabel, next_state=None):
        if next_state is None:
            try:
                return self._trans[state][ilabel]
            except KeyError:
                pass
            self._last_state += 1
            next_state = self._last_state
        self._trans[state][ilabel] = next_state
        return next_state

    def get_trans(self, state, ilabel):
        trans = self._trans.get(state)
        if trans is None:
            return
        return trans.get(ilabel)

    def iter_trans(self, state):
        for (ilabel, next_state) in self._trans[state].iteritems():
            yield (ilabel, next_state)

    def as_dot(self, name=None):
        if name is None:
            name = self.__class__.__name__
        dot = ['digraph {0} {{'.format(name)]
        dot.append('\trankdir=LR;')
        dot.append('\tnode [shape=circle];')
        dot.append('\tedge [arrowsize=.5];')
        todo = [self.start]
        seen = set()
        while todo:
            state = todo.pop()
            seen.add(state)
            if self.is_final(state):
                dot.append('{0} [shape=doublecircle label="{0}:{1}"];'.format(
                    state, self.get_olabel(state) or '&#949;'))
            for ilabel, next_state in self.iter_trans(state):
                dot.append('\t{0} -> {1} [style=solid label="{2}"];'.format(
                    state, next_state, ilabel.encode('utf-8')))
                if next_state not in seen:
                    todo.append(next_state)
                    seen.add(next_state)
        dot.append('}')
        return '\n'.join(dot)


class VerbMatcher(object):
    def __init__(self, verbs):
        self._trie = Trie()
        for verb in verbs:
            if '|' in verb:
                particle, stem = verb.split('|', 1)
            else:
                particle, stem = None, verb
            constraints = []
            if '[' in verb:
                stem, constr = stem.rstrip(']').rsplit('[', 1)
                for constraint in map(unicode.strip,
                        constr.rsplit('[', 1))[-1].split(','):
                    if '/' in constraint:
                        lemma, pos = constraint.split('/', 1)
                        constraints.append(partial(
                            self._has_pos_and_lemma, pos, lemma))
                    elif '@' in constraint:
                        lemma, dep = constraint.split('@', 1)
                        constraints.append(partial(
                            self._has_dep_and_lemma, dep, lemma))
                    else:
                        constraints.append(partial(self._has_lemma, lemma))
            if particle:
                self._trie.add_word(stem,
                    output=constraints[:] + [partial(
                        self._has_particle, particle)])
                self._trie.add_word('{0}{1}'.format(particle, stem),
                    output=constraints)
            else:
                self._trie.add_word(stem, output=constraints)

    def match(self, verb, dependants):
        try:
            constraints = self._trie.get(verb)
        except Trie.NotFinal:
            return False
        for constraint in constraints:
            if not any(constraint(dep) for dep in dependants):
                return False
        return True

    def _has_lemma(self, lemma, token):
        return token['lemma'] == lemma

    def _has_pos_and_lemma(self, pos, lemma, token):
        return token['pos'] == pos and token['lemma'] == lemma

    def _has_dep_and_lemma(self, dep, lemma, token):
        return token['deprel'] == dep and token['lemma'] == lemma

    def _has_particle(self, particle, token):
        return token['lemma'] == particle and token['pos'] == 'PTKVZ'


class StartOfClauseMatcher(object):
    class AlreadyFinalized(Exception):
        pass

    def __init__(self):
        self._forward_trie = Trie()
        self._reverse_trie = Trie()
        self._forward_depths = {}
        self._reverse_depths = {}
        self.finalized = False

    @classmethod
    def from_file(cls, filepath):
        matcher = cls()
        with codecs.open(filepath, encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                comment_pos = line.find('#')
                if comment_pos > -1:
                    line = line[:comment_pos].strip()
                if not line:
                    continue
                matcher.add_rule(line)
        matcher.finalize()
        return matcher

    def add_rule(self, rule):
        if self.finalized:
            raise self.AlreadyFinalized
        try:
            reverse_rule, forward_rule = rule.split('@', 1)
        except ValueError:
            reverse_rule, forward_rule = '', rule
        fstates = self._add(forward_rule.split('_'))
        rtokens = reverse_rule.split('_')
        rtokens.reverse()
        rstates = self._add(rtokens, reverse=True)
        for state in fstates:
            olabel = self._forward_trie.get_olabel(state)
            olabel.extend(rstates)
        for state in rstates:
            olabel = self._reverse_trie.get_olabel(state)
            olabel.extend(fstates)
        return self

    def finalize(self):
        if self.finalized:
            return self
        self._set_depths()
        self._set_depths(reverse=True)
        self.finalized = True
        return self

    def match(self, tokens, prev_tokens):
        self.finalize()
        if not hasattr(tokens, '__getitem__'):
            tokens = list(tokens)
        if not hasattr(prev_tokens, '__reverse__'):
            prev_tokens = list(prev_tokens)
        prev_tokens.reverse()
        fstack = [(self._forward_trie.start, 0)]
        rstack = [(self._reverse_trie.start, 0)]
        ffinals = {}
        rfinals = {}
        while fstack or rstack:
            if fstack:
                match = self._step(fstack, rstack, ffinals, rfinals, tokens)
                if match:
                    return True
            if rstack:
                match = self._step(rstack, fstack, rfinals, ffinals,
                                   prev_tokens, reverse=True)
                if match:
                    return True
        return False

    def _add(self, tokens, pos=0, prepend=None, state=None, reverse=False):
        if reverse:
            trie = self._reverse_trie
        else:
            trie = self._forward_trie
        if state is None:
            state = trie.start
        if prepend is not None:
            state = trie.add_trans(state, prepend)
        for idx, token in enumerate(tokens[pos:]):
            token = token.strip()
            if not token:
                continue
            if '|' in token:
                pos += idx + 1
                states = []
                for part in token.split('|'):
                    part = part.strip()
                    if not part:
                        continue
                    states.extend(self._add(tokens, pos=pos, prepend=part,
                                            state=state, reverse=reverse))
                return states
            state = trie.add_trans(state, token)
        trie.set_final(state, [])
        return [state] if state != trie.start else []

    def _set_depths(self, reverse=False):
        if reverse:
            trie = self._reverse_trie
            depths = self._reverse_depths
        else:
            trie = self._forward_trie
            depths = self._forward_depths
        todo = [(trie.start, 0)]
        while todo:
            state, depth = todo.pop()
            depths[state] = depth
            depth += 1
            for _, next_state in trie.iter_trans(state):
                todo.append((next_state, depth))

    def _step(self, stack1, stack2, finals1, finals2, tokens, reverse=False):
        if reverse:
            trie = self._reverse_trie
            depths = self._forward_depths
        else:
            trie = self._forward_trie
            depths = self._reverse_depths
        state, pos = stack1.pop()
        try:
            token = tokens[pos]
        except IndexError:
            return False
        pos += 1
        for attr in ('form', 'pos'):
            next_state = trie.get_trans(state, token[attr])
            if next_state is None:
                continue
            if pos < len(tokens):
                stack1.append((next_state, pos))
            if trie.is_final(next_state):
                seek_states = trie.get_olabel(next_state)
                if not seek_states:
                    return True
                if any(state in finals2 for state in seek_states):
                    return True
                elif any(state in finals2
                         for states in finals1.itervalues()
                            for state in states):
                    return True
                try:
                    min_depth = depths[stack2[-1][0]]
                except IndexError:
                    return False
                finals1[next_state] = [state for state in seek_states
                    if depths[state] >= min_depth]
        return False
