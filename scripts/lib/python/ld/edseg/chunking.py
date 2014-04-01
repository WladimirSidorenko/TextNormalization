#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from finitestateparsing import constraint, FiniteStateParser
from copy import deepcopy

import sys

##################################################################
def catgetter(token):
    return token['pos']

##################################################################
class UnificationFailure(Exception):
    pass

##################################################################
class FeatureMatrix(object):
    FEATS = [
        'nom',
        'acc',
        'dat',
        'gen',
        'sg',
        'pl',
        'masc',
        'fem',
        'neut',
    ]

    _FEAT_INDICES = dict((feat, idx) for (idx, feat) in enumerate(FEATS))

    def __init__(self, feats):
        bits = 0
        for feat in feats:
            idx = self._FEAT_INDICES.get(feat.lower())
            if idx is not None:
                bits |= 1 << idx
        if not bits & 0xf:
            bits |= 0xf
        if not (bits >> 4) & 0x3:
            bits |= 0x3 << 4
        if not (bits >> 6) & 0x7:
            bits |= 0x7 << 6
        self._bits = bits

    @classmethod
    def from_string(cls, feat_str):
        return cls([feat.strip() for feat in feat_str.split('|')])

    @classmethod
    def from_dict(cls, feat_dict):
        return cls([v for v in feat_dict.itervalues()])

    def unify(self, other):
        if not hasattr(other, '_bits'):
            return False
        bits = self._bits & other._bits
        if not self._unified(bits):
            raise UnificationFailure
        self._bits = bits
        return self

    def unifies(self, other):
        if not hasattr(other, '_bits'):
            return False
        return self._unified(self._bits & other._bits)

    def _unified(self, bits):
        return (bits & 0xf) and (bits >> 4) & 0x3 and (bits >> 6) & 0x7

    def __str__(self):
        return bin(self._bits)[2:]

##################################################################
class Chunker(object):
    def __init__(self):
        self._parser = FiniteStateParser()
        self._setup_parser()

    def chunk(self, sent):
        # convert word features to feature matrices
        # make a deep copy of sentence, in order not to use it destructively
        isent = deepcopy(sent)
        for token in isent:
            if token['pos'] in ('ART', 'NE', 'NN'):
                if isinstance(token['feat'], basestring):
                    token['feat'] = FeatureMatrix.from_string(token['feat'])
                elif isinstance(token['feat'], dict):
                    token['feat'] = FeatureMatrix.from_dict(token['feat'])
        return self._parser.parse(isent, catgetter=catgetter)

    def _setup_parser(self):
        add_rule = self._parser.add_rule

        add_rule('NC',
            '''
            <PPER>
            ''',
            level=1)

        @constraint
        def nc_month_spec_constraint(match):
            if match[2][0]['lemma'] not in ('Anfang', 'Mitte', 'Ende'):
                return False
            return match[3][0]['lemma'] in ('Januar',
                                            'Februar',
                                            'März',
                                            'Maerz',
                                            'April',
                                            'Mai',
                                            'Juni',
                                            'Juli',
                                            'August',
                                            'September',
                                            'Oktober',
                                            'November',
                                            'Dezember')

        add_rule('NC',
            '''
            (?:
                ^
            |
                [^<ART><CARD><PDAT><PDS><PIAT><PPOSAT>]
            )
            (
                (<NN>)
                (<NN>)
            )
            ''',
            constraint=nc_month_spec_constraint,
            group=1, level=1)

        @constraint
        def nc_det_noun_agreement(match):
            det = match[1]
            if not det:
                return True
            noun = match[2][0]
            try:
                if hasattr(noun['feat'], 'unify'):
                    noun['feat'].unify(det[0])
                else:
                    return False
            except UnificationFailure:
                return False
            return True

        add_rule('NC',
            '''
            (?:
                (<ART>)<PIAT>?
            |
                [<CARD><PDAT><PDS><PIAT><PPOSAT>]
            )?
            (?:
                (?:
                    <ADJA><ADJA>?
                )
                (?:
                    <$,>
                    (?:
                        <ADJA><ADJA>?
                    )
                )*
            )?
            ([<NE><NN>])
            ''',
            constraint=nc_det_noun_agreement,
            level=1)

        add_rule('NC',
            '''
            (?:
                <ART><PIAT>?
            |
                [<PDAT><PDS><PPOSAT><PIAT><CARD>]
            )
            <PC>
            <NC>
            ''',
            level=3)

        add_rule('PC',
            '''
            <APPR>   # preposition
            <NC>?
            <PDS>
            ''',
            level=2)

        @constraint
        def pc_genitive_adjunct_constraint(match):
            node = match[1][0]
            if node.last_child['pos'] != 'NN':
                return False
            art = node.first_child
            if art is None or art['pos'] != 'ART':
                return False
            if (not 'feats' in art) or (not hasattr(art['feats'], 'unifies')):
                return False
            return art['feats'].unifies(FeatureMatrix('gen'))

        add_rule('PC',
            '''
            [<APPR><APRRART>]
            <NC>
            (?:
                <KON>
                <NC>
            )*
            (<NC>)
            (?:
                <PDS>
            |
                [<APZR><PROAV>]    # optional circumposition or pronominal adverb
            )?
            ''',
            constraint=pc_genitive_adjunct_constraint,
            level=2)

        add_rule('PC',
            '''
            [<APPR><APPRART>]   # preposition
            <APPR>?             # optional embedded preposition ("bis an das Ende")
            (?:
                <ADV>           # adverbial chunk ("von damals")
                (?:             # optional conjunction
                    <KON>
                    <ADV>
                )?
            |
                <CARD>          # cardinal ("bis 1986")
                (?:             # optional conjunction
                    <KON>
                    <CARD>
                )?
            |
                <NC>            # noun chunk
                (?:             # optional conjunction
                    <KON>
                    <NC>
                )?
            )
            [<APZR><PROAV>]?    # optional circumposition or pronominal adverb
            ''',
            level=2)

        add_rule('AC',
            '''
            <ADV>*
            <PTKNEG>?
            <ADJD>+
            ''',
            level=3)

        add_rule('AC',
            '''
            <PTKA>?
            <ADV>+
            ''',
            level=3)

        add_rule('AC',
            '''
            <PROAV>
            ''',
            level=3)
