#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
from __future__ import unicode_literals

from conlex import ConLex
from adv_classification import AdvClassifier
from conll import CONLLWord

from bisect import bisect_left
from collections import defaultdict
from operator import itemgetter

import sys

##################################################################
EOC_PUNCT = ',;:.!?'
PRNT = 'parent'

##################################################################
def pairwise(iterable):
    '''Yield bigrams of an iterable.'''
    iterable = iter(iterable)
    prev = next(iterable, None)
    if prev is None:
        return
    yield None, prev
    for elem in iterable:
        yield prev, elem
        prev = elem

def get_index(a_tok):
    """Obtain index of token element."""
    return a_tok[0]

def get_string(a_tok):
    """Obtain index of token element."""
    return ' '.join([t.form for t in a_tok[-1]])

class EDS(object):
    '''Elementary discourse segment'''
    def __init__(self, tokens, type_, parent=None, prev=None, next=None):
        self.tokens = tokens  # Paula MarkNode
        self.parent = parent
        self.type = type_
        self.prev = prev
        self.next = next

    def __iter__(self):
        '''Yield the underlying tokens.'''
        for token in self.tokens:
            yield token

    def __getitem__(self, index):
        '''Return the token at the given index. Raises IndexError if
        the index exceeds the number of tokens.
        '''
        return self.tokens.ext[index]

    @property
    def pid(self):
        '''Return the EDS' PAULA ID.'''
        raise NotImplementedError
        return self.tokens.pid

    @property
    def is_embedded(self):
        return self.parent is not None

    @property
    def string(self):
        return ' '.join(get_string(tok) for tok in self)


class Connective(object):
    """Class representing a connective"""
    def __init__(self, tokens):
        self.tokens = tokens  # mark node
        self.sent = None  # mark node
        self.start = tokens[0]  # first token part
        self.end = tokens[-1]   # last token part
        self._int_arg = None
        self._ext_arg = None
        self.is_sent_initial = False
        self.is_para_initial = False
        self.syn_type = None
        self.arg_type = None
        self.maybe_anteponed = False
        self.maybe_postponed = False
        self.maybe_inserted = False
        self.relations = None
        self._string = u' '.join([get_string(tok) for tok in self])

    def __iter__(self):
        '''Yield the underlying tokens.'''
        for token in self.tokens:
            yield token

    def __getitem__(self, index):
        '''Return the token at the given index. Raises IndexError if
        the index exceeds the number of tokens.
        '''
        return self.tokens[index]

    @property
    def is_continuous(self):
        return True if len(self.tokens) == 1 else False

    @property
    def is_conjunction(self):
        return self.syn_type == 'conj'

    @property
    def is_subjunction(self):
        return self.syn_type == 'subj'

    @property
    def is_preposition(self):
        return self.syn_type == 'prep'

    @property
    def is_adverbial(self):
        return self.syn_type == 'adv'

    @property
    def is_complete(self):
        return self.int_arg and self.ext_arg

    @property
    def parts(self):
        return [self._get_string(tok) for tok in self]

    @property
    def pid(self):
        return self[0].pid

    @property
    def string(self):
        return self._string

    @property
    def int_arg(self):
        return self._int_arg

    @int_arg.setter
    def int_arg(self, arg):
        if hasattr(arg, 'ext'):
            arg = arg.ext
        self._int_arg = arg

    @property
    def int_start(self):
        if not self.int_arg:
            return
        return self.int_arg[0]

    @property
    def int_end(self):
        if not self.int_arg:
            return
        return self.int_arg[-1]

    @property
    def ext_arg(self):
        return self._ext_arg

    @ext_arg.setter
    def ext_arg(self, arg):
        if hasattr(arg, 'ext'):
            arg = arg.ext
        self._ext_arg = arg

    @property
    def ext_start(self):
        if not self.ext_arg:
            return
        return self.ext_arg[0]

    @property
    def ext_end(self):
        if not self.ext_arg:
            return
        return self.ext_arg[-1]

    def _get_string(self, a_tok):
        """Return string corresponding to given connective token."""
        return get_string(a_tok)

class ScopeFinder(object):
    """Class for finding spans of argument of discourse connectives."""

    def __init__(self, lex_path = None, adv_classifier = None,
                 greedy = True, rank_gram_func = True):
        """Initialize instance variables."""
        self._lex = ConLex(lex_path)
        self.adv_classifier = adv_classifier or AdvClassifier()
        self.greedy = greedy
        self.filter_chain = None
        self.rank_gram_func = rank_gram_func

    def find(self, forrest, eds_list):
        """
        Set the arguments of each connective in the given PAULA project.
        Connectives are read from the connective layer and assumed to be
        disambiguated.

        Note: As of now, this is actually not true.  Since there is no such
        layer yet, connectives are identified by this class.  No disambiguation
        is attempted, though.

        """
        self._initialize(forrest, eds_list)

        for con in self._connectives:
            if con.arg_type == 'intra_sent':
                self._handle_intra_sent_connective(con)
            else:
                self._handle_inter_sent_connective(con)

    def _initialize(self, forrest, sds_list):
        # conll_forrest is a collection of multiple CONLL sentences
        self._forrest = forrest
        # list of sentences belonging to paragraph
        self._sentences = self._forrest.sentences
        # we assume that only the first sentence of every tweet introduces a
        # new paragraph
        self._para_starts = set([0])
        # list of sentences belonging to paragraph
        self._tokens = [((s_id, t_id), [tok]) \
                            for s_id, snt in enumerate(self._sentences) \
                            for tok, t_id in snt.get_words()]
        self._max_pos = len(self._tokens)
        # `_t_id2pos' dict will allow us to access individual tokens in token
        # list by their indices in constant time
        self._t_id2pos = dict((tok_id, pos) for pos, (tok_id, tok) in enumerate(self._tokens))
        # remember EDS's
        self._eds, self._eds_ends = self._load_eds(sds_list)
        # iterate over connectors
        self._connectives = []
        for con in self._iter_connectives():
            con = Connective(con)
            con.sent_index = self._get_sent_index(con.start)
            con.sent = self._sentences[con.sent_index]
            con.is_sent_initial = self._is_sent_initial(con)
            con.is_para_initial = self._is_para_initial(con)
            con.syn_type = self._lex.type(con.string)
            con.arg_type = self._determine_arg_type(con)
            con.maybe_anteponed = self._lex.maybe_anteponed(con.string)
            con.maybe_postponed = self._lex.maybe_postponed(con.string)
            con.maybe_inserted = self._lex.maybe_inserted(con.string)
            con.relations = self._lex.relations(con.string)
            self._connectives.append(con)
        if self.filter_chain is not None:
            self.filter_chain.initialize()

    def _determine_arg_type(self, con):
        # Return on of the types 'intra_sent' or 'inter_sent'
        syn_type = con.syn_type
        if syn_type == 'adv':
            if con.is_sent_initial:
                return self.adv_classifier(con, self)
            return 'intra_sent'
        elif syn_type == 'conj':
            if con.is_sent_initial:
                return 'inter_sent'
            return 'intra_sent'
        elif syn_type == 'prep':
            return 'intra_sent'
        elif syn_type == 'subj':
            return 'intra_sent'

    def _handle_inter_sent_connective(self, con):
        # Handle connectives whose external argument is within the same
        # sentence as the connective.
        if con.is_para_initial:
            ext_arg = self._get_preceding_sent(con)
        else:
            sents = self._get_candidate_sents(con)
            if len(sents) == 1:
                ext_arg = sents.pop()[1]
            else:
                if self.filter_chain is not None:
                    sents = self.filter_chain(sents, con)
                num_sents = len(sents)
                if num_sents == 1:
                    ext_arg = sents.pop()[1]
                else:
                    ext_arg = self._rank_sents(sents, con)
        con.int_arg = self._get_eds_by_start(con.start)
        con.ext_arg = ext_arg

    def _handle_intra_sent_connective(self, con):
        # Handle connectives whose external argument is located in a
        # different sentence than the connective.
        # print >> sys.stderr, "con = ", repr([get_string(t) for t in con.tokens])
        if con.is_adverbial:
            # print >> sys.stderr, "Con is adverbial."
            self._handle_adverbial(con)
        elif con.is_conjunction:
            # print >> sys.stderr, "Con is conjunction."
            self._handle_conjunction(con)
        elif con.is_subjunction:
            # print >> sys.stderr, "Con is subjunction."
            self._handle_subjunction(con)
        else:
            # print >> sys.stderr, "Con is preposition."
            self._handle_preposition(con)
        if not con.is_complete and not con.is_adverbial:
            # print >> sys.stderr, "Con is not complete and not adverbial."
            # print >> sys.stderr, "(before) con.int_arg = ", repr(con.int_arg)
            con.int_arg = self._ortho_find_int_arg(con)
            # print >> sys.stderr, "(after) con.int_arg = ", repr(con.int_arg)
            # print >> sys.stderr, "(before) con.ext_arg = ", repr(con.ext_arg)
            con.ext_arg = self._ortho_find_ext_arg(con)
            # print >> sys.stderr, "(after) con.ext_arg = ", repr(con.ext_arg)

    def _handle_adverbial(self, con):
        if not con.is_continuous:
            # print >> sys.stderr, "kon is not continuous"
            con.int_arg = self._get_eds_by_start(con.start)
            # print >> sys.stderr, "con.int_arg =", repr(con.int_arg)
            con.ext_arg = self._get_eds_by_start(con.end)
            # print >> sys.stderr, "con.ext_arg =", repr(con.ext_arg)
            # Internal and external arguments are in the same segment.
            if con.int_arg == con.ext_arg:
                # print >> sys.stderr, "splitting discontinuous connective"
                self._split_discontinuous_connective(con)
            return
        # otherwise, we assume that the connective consists of only one word
        if not self._is_clause_initial(con) and not self._is_clause_final(con):
            # print >> sys.stderr, "con is clause initial and not clause final"
            con.int_arg = con.sent
            con.ext_arg = self._get_preceding_sent(con.start)
            return
        con.int_arg = self._get_eds_by_start(con.start)
        arg = []
        # print >> sys.stderr, "con.int_arg =", repr(con.int_arg)
        eds = con.int_arg.prev
        # print >> sys.stderr, "Entering loop"
        while True:
            if eds is None:
                # print >> sys.stderr, "eds is None"
                break
            if self.greedy:
                # print >> sys.stderr, "appending arg"
                arg.append(eds)
            if eds.type == 'MainCl':
                # print >> sys.stderr, "eds type is MainCl"
                if not self.greedy:
                    arg.append(eds)
                arg.reverse()
                con.ext_arg = [tok for part in arg for tok in part]
                break
            # print >> sys.stderr, "appending eds to prev"
            eds = eds.prev

    def _handle_conjunction(self, con):
        # print >> sys.stderr, "con.start = ", con.start
        con.int_arg = self._get_eds_by_start(con.start)
        # print >> sys.stderr, "con.int_arg = ", repr(con.int_arg)
        if con.maybe_inserted:
            # print >> sys.stderr, "con may be inserted"
            if con.int_arg.is_embedded:
                # print >> sys.stderr, "con.int_arg.is_embedded"
                con.ext_arg = self._get_eds_by_id(con.int_arg.parent)
                return con
        elif self._is_clause_initial(con):
            # print >> sys.stderr, "con is clause initial"
            # It's postponed.
            self._find_ext_arg_of_postponed_subj(con)
            if con.is_complete:
                return con

    def _handle_subjunction(self, con):
        # print >> sys.stderr, "handling subjunction", repr(con.tokens)
        int_arg = self._get_eds_by_start(con.start)
        # print >> sys.stderr, "int_arg = ", repr(int_arg)
        if int_arg.type != 'SubCl':
            return
        con.int_arg = int_arg
        if con.is_continuous:
            if con.maybe_inserted:
                if con.int_arg.is_embedded:
                    con.ext_arg = self._get_eds_by_id(con.int_arg.parent)
                    return con
            if con.maybe_postponed and not con.is_sent_initial:
                self._find_ext_arg_of_postponed_subj(con)
                if con.is_complete:
                    return con
            if con.maybe_anteponed and con.is_sent_initial:
                self._find_ext_arg_of_anteponed_subj(con)
                if con.is_complete:
                    return con
        else:
            con.ext_arg = self._get_eds_by_start(con.end)
            # Internal and external arguments are in the same segment.
            if con.int_arg == con.ext_arg:
                self._split_discontinuous_connective(con)
            else:
                self._complete_discontinuous_int_arg(con)

    def _handle_preposition(self, con):
        con.int_arg = self._get_eds_by_start(con.start)
        if con.int_arg.parent is None:
            # Something went wrong, we don't have a preposition at hand.
            con.int_arg = self._make_preposition_chunk(con)
            con.ext_arg = self._get_sent(con.start)
        else:
            con.ext_arg = self._get_eds_by_id(con.int_arg.parent)

    def _split_discontinuous_connective(self, con):
        # Split a discontinuous connective by orthographic clues such that
        # internal and external arguments are directly adjacent to one another.
        # The start of the internal argument corresponds to the next left
        # clause boundary relative to the argument.  The end of the external
        # corresponds to the next right clause boundary of the argument.
        int_arg = []
        # print >> sys.stderr, "con.start = ", repr(con.start)
        for w in con.start[-1][::-1]:
            int_arg.append(w)
        # print >> sys.stderr, "int_arg = ", repr(int_arg)
        for tok in self._iter_sent_from(con.start, reverse=True):
            # print >> sys.stderr, "appending tok =", repr(tok.form)
            int_arg.append(tok)
            if tok.form in EOC_PUNCT:
                break
        int_arg.reverse()
        end = con.end[-1][-1]
        for tok in self._iter_sent_from(con.start):
            if tok == end:
                break
            int_arg.append(tok)
        ext_arg = []
        for w in con.end[-1]:
            ext_arg.append(w)
        for tok in self._iter_sent_from(con.end):
            ext_arg.append(tok)
            if tok.form in EOC_PUNCT:
                break
        con.int_arg = int_arg
        con.ext_arg = ext_arg

    def _complete_discontinuous_int_arg(self, con):
        # Complete the internal argument of a connective by extending
        # it by all EDS up to one holding the external argument.
        eds = con.int_arg
        arg = []
        while True:
            if eds is None or eds == con.ext_arg:
                break
            arg.extend(*eds)
            eds = eds.next
        con.int_arg = arg

    def _make_preposition_chunk(self, con):
        # Perform a (very) basic chunking of prepositional phrases.
        tokens = self._iter_sent_from(con.start)
        if con.is_continuous:
            arg = []
            for tok in tokens:
                arg.append(tok)
                if self._get_pos(tok).startswith('N'):
                    tok = next(tokens, None)
                    # Check for a genetive complement.
                    if self._has_case(tok, 'gen'):
                        arg.append(tok)
                        for tok in tokens:
                            arg.append(tok)
                            if self._get_pos(tok).startswith('N'):
                                break
                    break
            return arg
        # The connective is dicontinuous. Finding the NP is trivial
        # since it's located directly between preposition and
        # circumfix.
        arg = []
        for tok in tokens:
            arg.append(tok)
            # print >> sys.stderr, "tok =", repr(tok)
            # print >> sys.stderr, "con.end =", repr(con.end)
            if tok.form == get_string(con.end):
                break
        return arg

    def _has_case(self, tok, case):
        # Return true if the given token's case and the given case
        # agree, false otherwise.
        if tok is None:
            return False
        if case == '*':
            return True
        feats = tok.feat or tok.pfeat
        # print >> sys.stderr, repr(feats)
        # sys.exit(66)
        try:
            case_feat = feats["case"]
        except KeyError:
            return False
        return case_feat == '*' or case_feat.lower() == case.lower()

    def _find_ext_arg_of_postponed_subj(self, con):
        # Find the external argument of a postponed subjunction.
        # Performs a search to the left until the next main clause is
        # hit. If greediness is on, all intervening clauses will be
        # added to the argument.
        arg = []
        eds = con.int_arg.prev
        while True:
            if eds is None:
                break
            if self.greedy:
                arg.append(eds)
            if eds.type == 'MainCl':
                if not self.greedy:
                    arg.append(eds)
                arg.reverse()
                con.ext_arg = [tok for part in arg for tok in part]
                return
            eds = eds.prev

    def _find_ext_arg_of_anteponed_subj(self, con):
        # Find the external argument of an anteponed subjunction.
        # Performs a search to the right until the next main clause is
        # hit. If greediness is on, all intervening clauses will be
        # added to the argument.
        arg = []
        eds = con.int_arg.next
        while True:
            if eds is None:
                break
            elif self.greedy:
                arg.extend(eds)
            if eds.type == 'MainCl':
                try:
                    verb = eds[1]
                except IndexError:
                    return
                if self._get_pos(verb).startswith('V'):
                    if not self.greedy:
                        arg.extend(eds)
                    else:
                        eds = eds.next
                        while True:
                            if eds is None or eds.type == 'MainCl':
                                break
                            else:
                                arg.extend(eds)
                            eds = eds.next
                    con.ext_arg = arg
                    return
                break
            eds = eds.next

    def _ortho_find_int_arg(self, con):
        # Find the internal argument of a connective by means of
        # orthography-based heuristics.
        if con.is_conjunction:
            prev_tok = self._get_prev_token(con.start)
            next_tok = self._get_next_token(con.start)
            if prev_tok is not None and next_tok is not None:
                prev_pos = self._get_pos(prev_tok)
                next_pos = self._get_pos(next_tok)
                for pos in ('AD', 'V', 'N'):
                    if prev_pos.startswith(pos) and next_pos.startswith(pos):
                        # Conjunction below the clause level, ignore.
                        return
                if prev_pos.startswith('APPR') and next_pos.startswith('APPR'):
                    arg = []
                    for tok in self._iter_sent_from(next_tok):
                        arg.append(tok)
                        if self._get_pos(tok).startswith('N'):
                            break
                    return arg
            arg = []
            for tok in self._iter_sent_from(con.start):
                arg.append(tok)
                if tok.form in EOC_PUNCT:
                    break
            return arg
        # print >> sys.stderr, "con.tokens =", repr(con.tokens)
        arg = [t for tok_tuple in con.tokens for t in tok_tuple[-1]]
        # print >> sys.stderr, "arg =", repr(arg)
        if con.is_sent_initial:
            for prev_tok, tok in pairwise(self._iter_sent_from(con.start)):
                form = get_string(tok)
                if form in ';:' or (form == ',' and self._is_verb(prev_tok)):
                    break
                arg.append(tok)
            return arg
        for prev_tok, tok in pairwise(self._iter_sent_from(con.start)):
            # print >> sys.stderr, "tok =", repr(tok)
            form = tok.form
            if form in EOC_PUNCT or (form == ',' and self._is_verb(prev_tok)):
                break
            arg.append(tok)
        # print >> sys.stderr, "return arg =", repr(arg)
        return arg

    def _ortho_find_ext_arg(self, con):
        # Find the external argument of a connective by means of
        # orthography-based heuristics.
        if con.is_conjunction:
            prev_tok = self._get_prev_token(con.start)
            next_tok = self._get_next_token(con.start)
            # check that surrounding tokens are not None
            if (prev_tok is not None) and (next_tok is not None):
                prev_pos = self._get_pos(prev_tok)
                next_pos = self._get_pos(next_tok)
                for pos in ('AD', 'APPR', 'V', 'N'):
                    if prev_pos.startswith(pos) and next_pos.startswith(pos):
                        # print >> sys.stderr, "[prev_tok]"
                        return prev_tok[-1]
                # print >> sys.stderr, "Taking until start of clause (I)."
                return self._take_until_start_of_clause(con.start)
        if con.is_sent_initial:
            # connective is anteponed
            # print >> sys.stderr, "Taking until end of clause."
            return self._take_until_end_of_clause(con.int_end)
        # assume connective is postponed
        # print >> sys.stderr, "Taking until start of clause (II)."
        return self._take_until_start_of_clause(con.start)

    def _take_until_start_of_clause(self, tok):
        tokens = []
        first_punct = True
        for tok in self._iter_sent_from(tok, reverse=True):
            tokens.append(tok)
            if tok.form in EOC_PUNCT:
                if first_punct:
                    first_punct = False
                else:
                    break
        tokens.reverse()
        # print >> sys.stderr, "_take_until_start_of_clause: tokens =", repr(tokens)
        return tokens

    def _take_until_end_of_clause(self, tok):
        tokens = []
        first_punct = True
        for tok in self._iter_sent_from(tok):
            tokens.append(tok)
            if get_string(tok) in EOC_PUNCT:
                if first_punct:
                    first_punct = False
                else:
                    break
        # print >> sys.stderr, "_take_until_start_of_clause: tokens =", repr(tokens)
        return tokens

    def _get_sent_index(self, tok):
        """Return index of the sentence to which given token belongs."""
        return tok[0][0]

    def _get_sent(self, tok):
        """Obtain sentence pertaining to particular token."""
        return self._sentences[self._get_sent_index(tok)]

    def _get_pos(self, tok):
        if hasattr(tok, "pos"):
            return tok.pos
        else:
            return tok[-1][0].pos

    def _get_deprel(self, tok):
        return tok.pdeprel

    def _is_verb(self, tok):
        if tok is None:
            return False
        return self._get_pos(tok).startswith('V')

    def _get_preceding_sent(self, tok):
        """Get sentence preceding the connector."""
        # print >> sys.stderr, repr(tok)
        idx = self._get_sent_index(tok)
        if idx == 0:
            return None
        # print >> sys.stderr, repr(idx)
        return self._sentences[idx - 1]

    def _iter_sent(self, tok, reverse=False):
        sent = self._get_sent(tok)
        if reverse:
            sent = reversed(sent)
        for tok in sent:
            yield tok

    def _iter_sent_from(self, tok, reverse = False):
        t_id = tok[0][1]
        tokens = [t for t in self._iter_sent(tok)]
        if reverse:
            tokens = reversed(tokens[:t_id])
        else:
            tokens = tokens[t_id + 1:]
        for t in tokens:
            yield t

    def _get_candidate_sents(self, con):
        sents = []
        # print >> sys.stderr, "con.sent_index", repr(con.sent_index)
        for idx in xrange(con.sent_index - 1, -1, -1):
            sent = self._sentences[idx]
            # print >> sys.stderr, "sent", repr(sent)
            sents.append((idx, sent))
            if idx in self._para_starts:
                break
        return sents

    _GRAM_FUNC_ORDER = ('SB', 'OA', 'OA2', 'DA', 'OG')

    def _rank_sents(self, sents, con):
        # print >> sys.stderr, "sents are:", repr(sents)
        # print >> sys.stderr, "con.sent is", repr(con.sent)
        sent = con.sent         # CONLL sentence
        sent_indices = [sent_idx for sent_idx, _ in sents]
        # print >> sys.stderr, "sent_indices =", repr(sent_indices)
        seen = set()
        if self.rank_gram_func:
            sent = sorted(sent, key=lambda tok: self._get_rank(tok, sent))
        # currently deactivated because we don't have coref chains
        # for tok in sent:
        #     pos = tok.pos
        #     if pos not in ('NE', 'NN', 'PPER'):
        #         continue
        #     strng = tok.form
        #     if strng in seen:
        #         continue
        #     seen.add(strng)
        #     coref_chains = self._find_coref_chains(tok, sent_indices)
        #     if not coref_chains:
        #         continue
        #     winner = self._apply_coref_rules(coref_chains, pos)
        #     if winner is not None:
        #         return winner
        return self._get_preceding_sent(con[0])

    def _get_rank(self, tok, sent):
        try:
            return self._GRAM_FUNC_ORDER.index(self._get_deprel(tok))
        except ValueError:
            return int(tok.idx) + len(sent)

    def _get_prev_token(self, tok):
        """Obtain token preceding the tok."""
        # obtain id of the token
        t_id = tok[0]
        # print >> sys.stderr, "t_id = ", t_id
        # obtain position of this token in token list
        prev_pos = self._t_id2pos[t_id] - 1
        if prev_pos < 0:
            return None
        else:
            return self._tokens[prev_pos]

    def _get_next_token(self, tok):
        """Obtain token following the tok."""
        # obtain id of the token
        t_id = tok[0]
        # obtain position of this token in the token list
        next_pos = self._t_id2pos[t_id] + 1
        if next_pos < self._max_pos:
            return self._tokens[next_pos]
        else:
            return None

    def _is_para_initial(self, con):
        """Check if connective is located at the beginning of first sentence."""
        return con[0][0] == 0

    def _is_sent_initial(self, con):
        """Check if connective is located at the beginning of a sentence."""
        return con[0][1] == 0

    def _in_simplex_sent(self, con):
        return all(get_string(tok) not in ',;:' for tok in con.sent.ext)

    def _is_clause_initial(self, con):
        # print >> sys.stderr, "(_is_clause_initial) con = ", repr(con)
        tok = con.start
        for _ in xrange(2):
            tok = self._get_prev_token(tok)
            # if no previous token exists or if this token is a punctuation
            # mark, return True
            if tok is None or get_string(tok) in '.,;:':
                return True
        return False

    def _is_clause_final(self, con):
        tok = con.end
        next_tok = self._get_next_token(tok)
        # print >> sys.stderr, "next_tok = ", repr(next_tok)
        return next_tok is None or get_string(next_tok) in ',;:.!?'

    def _load_eds(self, a_sds_list):
        """Load elementary discourse segments from list."""
        # print >> sys.stderr, "a_sds_list =", repr(a_sds_list)
        eds_list = []
        eds_ends = []
        last_eds = curr_eds = eds = None
        last_sent = None
        edstype = None
        # each sds tree in the a_sds_list corresponds to a consecutive input
        # sentence
        for sent_i, sds_tree in enumerate(a_sds_list):
            # iterate over each EDS in this sentence
            last_eds = None
            for eds in sds_tree:
                # eds.pretty_print()
                if eds.last_terminal is None:
                    continue
                # edstype = eds.feats['type']
                if PRNT in eds.feats:
                    eds_list.append(EDS(eds, edstype, parent = eds.feats[PRNT]))
                else:
                    curr_eds = EDS(eds, edstype, prev = last_eds)
                    eds_list.append(curr_eds)
                    if last_eds is not None:
                        last_eds.next = curr_eds
                    last_eds = curr_eds
                # append index of the last word in given EDS to eds_ends
                eds_ends.append((sent_i, int(eds.last_terminal.idx) - 1))
        # print >> sys.stderr, "eds_list =", repr(eds_list)
        return eds_list, eds_ends

    def _get_eds_by_start(self, tok):
        # print >> sys.stderr, "get_index(tok) = ", get_index(tok)
        # print >> sys.stderr, "self._eds_ends = ", repr(self._eds_ends)
        # print >> sys.stderr, "bisect_left(self._eds_ends, get_index(tok)) = ", \
        #     bisect_left(self._eds_ends, get_index(tok))
        return self._eds[bisect_left(self._eds_ends, get_index(tok))]

    def _get_eds_by_id(self, eds_id):
        raise NotImplementedError
        layer = self._get_paula_layer('edseg.eds')
        return layer.idmap[eds_id]

    def _iter_connectives(self):
        # This is a temporary kludge. (Disambiguated) Connectives will be given
        # by the Paula project.
        from _conn_ident import ConnectiveFinder
        finder = ConnectiveFinder(self._lex)
        # print >> sys.stderr, "self._tokens = ", repr(self._tokens)
        connectives = sorted(finder.find(self._tokens), key = lambda x: x[0][0])
        # print >> sys.stderr, "connectives = ", repr(connectives)
        # Sort connectives in document order.
        # print [(c[1].getString(), c[1].ext[0].start) for c in connectives]
        for con in connectives:
            yield con

    def pretty_print(self, a_stream = sys.stdout, a_encoding = None):
        """Output connectives with their arguments in a readable way."""
        con_id = 0
        arg_id = 0
        ostring = u""
        for con in self._connectives:
            # print >> sys.stderr, "con =", repr(con)
            # print >> sys.stderr, "con.is_complete =", repr(con.is_complete)
            if not con.is_complete:
                continue
            ostring += u'("' + ' '.join([w.form for _id, t in con.tokens for w in t]) + u'"\n'
            # print >> sys.stderr, repr(con.int_arg)
            ostring += u"\t(intern:\n" + self._arg2str(con.int_arg) + u'\t)\n'
            # print >> sys.stderr, repr(con.ext_arg)
            ostring += u"\t(extern:\n" + self._arg2str(con.ext_arg) + u"\t))\n"
        if a_encoding:
            ostring = ostring.encode(a_encoding)
        a_stream.write(ostring)

    def _arg2str(self, a_arg, a_indent = '\t', a_lvl = 2):
        """Convert CONLL Sentence to string."""
        ret = u""
        indent = a_indent * a_lvl
        w = None
        # print >> sys.stderr, repr(a_arg)
        # arg can be either CONLL sentence or an EDS
        for w in a_arg:
            if isinstance(w, CONLLWord):
                ret += indent + u"{form:s}/{tag:s}".format(form = w.form, tag = w.pos) + u'\n'
            else:
                for line in str(w).splitlines():
                    ret += indent + line
        return ret
