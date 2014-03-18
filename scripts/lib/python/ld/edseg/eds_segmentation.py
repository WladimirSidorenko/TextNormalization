#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from clause_segmentation import ClauseSegmenter
from data import data_dir
import data as data
from finitestateparsing import Tree
from util import StartOfClauseMatcher, Trie


def pairwise(iterable):
    iterable = iter(iterable)
    prev = next(iterable, None)
    if prev is None:
        return
    yield None, prev
    for elem in iterable:
        yield prev, elem
        prev = elem


class EDSSegmenter(object):
    SDS_LABEL = 'SDS'
    EDS_LABEL = 'EDS'
    MAIN_CLAUSE = 'MainCl'
    SUB_CLAUSE = 'SubCl'
    REL_CLAUSE = 'RelCl'
    PAREN = 'Paren'
    DISCOURSE_PP = 'DiPP'

    def __init__(self, **kwargs):
        clause_segmenter = kwargs.get('clause_segmenter')
        if clause_segmenter is None:
            self._clause_segmenter = ClauseSegmenter()
        else:
            self._clause_segmenter = clause_segmenter
        self._clause_discarder = StartOfClauseMatcher.from_file(
            data_dir('skip_rules.txt'))
        self._sent = None
        self._tokens = []

    def segment(self, sent):
        self._sent = sent
        clauses = self._clause_segmenter.segment(sent)
        sds = Tree(self.SDS_LABEL)
        eds = self._make_eds(sds, type=self.MAIN_CLAUSE)
        for idx, clause in enumerate(clauses):
            eds = self._process_clause(clause, idx, clauses, sds, eds)
        return sds

    def _process_clause(self, clause, idx, clauses, sds, eds, depth=0):
        if not clause:
            return eds
        clause = self._flatten_coord(clause)
        if self._is_embedded(idx, clauses):
            depth += 1
        child1 = clause.first_child
        if not self._is_token(child1) and child1.label == clause.label:
            for idx, child in enumerate(clause):
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
            return eds
        self._tokens, prev_toks = list(clause.iter_terminals()), self._tokens
        if self._clause_discarder.match(self._tokens, prev_toks):
            for idx, child in enumerate(clause):
                if self._is_token(child):
                    eds.append(child)
                else:
                    eds = self._process_clause(child, idx, clause, sds, eds,
                                               depth=depth)
            return eds
        try:
            meth = getattr(self, '_process_{0}'.format(clause.label.lower()))
        except AttributeError:
            return eds
        return meth(clause, idx, clauses, sds, eds, depth=depth)

    def _process_maincl(self, clause, idx, parent, sds, eds, depth=0):
        verb, deps = self._find_verb_and_dependants(clause)
        if verb is None:
            self._flatten(clause)
            eds.extend(clause)
            return eds
        elif (len(eds) and
              not data.reporting_verbs.match(verb['lemma'], deps) and
              not self._is_unintroduced_complement(clause, idx, parent)):
            if depth > 0:
                eds = self._make_embedded_eds(eds, type=self.MAIN_CLAUSE)
            elif len(eds):
                eds = self._make_eds(sds, type=self.MAIN_CLAUSE)
        first_token = True
        for idx, child in enumerate(clause):
            if self._is_token(child):
                if first_token:
                    if eds.get('type') != self.MAIN_CLAUSE:
                        eds = self._make_eds(sds, type=self.MAIN_CLAUSE)
                    first_token = False
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return sds.last_child

    def _process_sntsubcl(self, clause, idx, parent, sds, eds, depth=0):
        is_complement = False
        for prev, token in pairwise(clause.terminals(3)):
            if not self._is_token(token):
                break
            elif token and token['lemma'] in ('dass', 'daß', 'ob'):
                if prev is None or (prev['pos'] != 'ADV' and prev['lemma'] != 'so'):
                    is_complement = True
                break
        if not is_complement:
            if depth > 0:
                eds = self._make_embedded_eds(eds)
            elif len(eds):
                eds = self._make_eds(sds)
            eds.set('type', self.SUB_CLAUSE)
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return sds.last_child

    def _process_infsubcl(self, clause, idx, parent, sds, eds, depth=0):
        if self._is_nonfin_subord(clause, eds):
            if len(eds):
                eds = self._make_eds(sds)
            eds.set('type', self.SUB_CLAUSE)
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return eds

    def _process_relcl(self, clause, idx, parent, sds, eds, depth=0):
        if depth > 0:
            eds = self._make_embedded_eds(eds)
        elif len(eds):
            eds = self._make_eds(sds)
        eds.set('type', self.REL_CLAUSE)
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return sds.last_child

    def _process_infcl(self, clause, idx, parent, sds, eds, depth=0):
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return eds

    def _process_intcl(self, clause, idx, parent, sds, eds, depth=0):
        child = None
        if clause.terminal(1)['form'] in ('weshalb', 'weswegen'):
            if depth > 0:
                eds = self._make_embedded_eds(eds)
            elif len(eds):
                eds = self._make_eds(sds)
            eds.set('type', self.SUB_CLAUSE)
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        if self._is_token(child) and child['form'] == ':':
            eds = self._make_eds(sds)
        return eds

    def _process_paren(self, clause, idx, parent, sds, eds, depth=0):
        if clause.first_terminal['form'] not in data.QUOTES and \
                any(tok['pos'].startswith('VV')
                    for tok in clause.iter_terminals()):
            if depth > 0:
                eds = self._make_embedded_eds(eds, type=self.PAREN)
            elif len(eds):
                eds = self._make_eds(sds, type=self.PAREN)
        eds.extend(clause)
        return sds.last_child

    def _process_pc(self, clause, idx, parent, sds, eds, depth=0):
        tokens = list(clause.iter_terminals())
        try:
            circumpos = data.discourse_preps.get(
                ' '.join(tok['form'] for tok in tokens), holistic=False)
        except Trie.NotFinal:
            eds.extend(tokens)
            return eds
        if circumpos is None or tokens[-1]['form'] == circumpos:
            eds = self._make_embedded_eds(eds, type=self.DISCOURSE_PP)
            eds.extend(tokens)
            return sds.last_child
        token1 = parent[idx + 1]
        if token1 is None or not self._is_token(token1):
            eds.extend(tokens)
            return eds
        if token1 is None or token1['form'] != circumpos:
            eds.extend(tokens)
            return eds
        token2 = parent[idx + 2]
        if token2 is None or (self._is_token(token2) and
                              token2['pos'] in ('$,', '$.', '$(', 'ADJD',
                                                'ADV', 'APPR', 'VVPP')):
            parent.remove(token1)
            tokens.append(token1)
            eds = self._make_embedded_eds(eds, type=self.DISCOURSE_PP)
        eds.extend(tokens)
        return sds.last_child

    def _process_any(self, clause, idx, parent, sds, eds, depth=0):
        if eds is None:
            eds = self._make_eds(sds, self.MAIN_CLAUSE)
        self._flatten(clause)
        eds.extend(clause)

    def _is_unintroduced_complement(self, clause, idx, parent):
        prev_clause = parent[idx - 1] if idx else None
        if prev_clause is None:
            return False
        prev_verb, deps = self._find_verb_and_dependants(prev_clause)
        if prev_verb is None:
            return False
        tokens = clause.iter_terminals()
        token1 = next(tokens, {})
        if token1.get('pos') == 'KON':
            return False
        # Test for cases like "..., nämlich dass ..."
        elif (token1.get('pos') != 'ADV' and
              next(tokens, {}).get('lemma') in ('dass', 'daß')):
            return False
        return data.dass_verbs.match(prev_verb['lemma'], deps)

    def _is_nonfin_subord(self, clause, eds):
        # Test for cases like 'zu <adj> ..., um ... (to <adj> to ...)'
        if not any(tok['lemma'] in 'um' for tok in clause.children(2)):
            return True
        tokens = list(eds.iter_terminals())[-1:-10:-1]
        try:
            for idx, token in enumerate(tokens):
                if token['pos'] in ('ADJD', 'VVPP'):
                    if (tokens[idx - 1]['lemma'] == 'genug' or
                        tokens[idx + 1]['lemma'] in ('genug', 'zu')):
                        return False
        except IndexError:
            pass
        return True

    def _flatten(self, node, parent=None):
        if self._is_token(node):
            return
        for child in node:
            self._flatten(child, parent=node)
        if parent is not None:
            parent.replace(node, *node)

    def _flatten_coord(self, clause):
        if self._is_token(clause):
            return clause
        try:
            child1, child2 = clause.children(2)
        except ValueError:
            return clause
        if self._is_token(child1) or child1.label != clause.label:
            return clause
        elif not self._is_token(child2) or child2['pos'] != 'KON':
            return clause
        new_clause = Tree(clause.label)
        pending_conj = None
        for child in clause:
            if self._is_token(child):
                assert child['pos'] == 'KON'
                pending_conj = child
            elif pending_conj is not None:
                conjunct = child
                child1 = conjunct.first_child
                while not self._is_token(child1):
                    conjunct = child1
                    child1 = child1.first_child
                conjunct.insert(0, pending_conj)
                pending_conj = None
                new_clause.append(child)
            else:
                new_clause.append(child)
        return new_clause

    def _is_token(self, node):
        return node is not None and not isinstance(node, Tree)

    def _is_embedded(self, idx, parent):
        if self._is_token(parent[idx]) or not idx:
            return False
        return (self._is_token(parent[idx - 1]) and
                self._is_token(parent[idx + 1]))

    def _find_verb_and_dependants(self, clause):
        verb = clause.get('verb')
        if verb is None:
            return None, []
        deps = []
        for token in clause.iter_terminals():
            if token['head'] == verb['idx']:
                deps.append(token)
        return verb, deps

    def _make_eds(self, sds, **feats):
        sds.append(Tree(self.EDS_LABEL, feats=feats))
        return sds.last_child

    def _make_embedded_eds(self, eds, **feats):
        feats['embedded'] = True
        embed_eds = Tree(self.EDS_LABEL, feats=feats)
        eds.append(embed_eds)
        return embed_eds
