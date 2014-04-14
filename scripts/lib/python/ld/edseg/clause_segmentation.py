#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import data
from chunking import Chunker
from finitestateparsing import FiniteStateParser, Tree
from util import match as match_

import sys

def print_feat((feat, val)):
    if isinstance(val, dict):
        sane_val = val.get('lemma')
    else:
        sane_val = val
    return u'{0}={1}'.format(feat, sane_val)

def catgetter(node):
    if hasattr(node, 'label'):
        return node.label
    form = unicode(node['form'])
    if form in data.DELIM_NAMES:
        return data.DELIM_NAMES[form]
    return node['pos']

class ClauseSegmenter(object):
    def __init__(self, **kwargs):
        chunker = kwargs.get('chunker')
        if chunker is None:
            self._chunker = Chunker()
        else:
            self._chunker = chunker
        self._parser = FiniteStateParser()
        self._setup_parser()

    def segment(self, sent):
        # print >> sys.stderr, "Preparing tokens..."
        self._prepare_tokens(sent)
        # print >> sys.stderr, "Tokens prepared..."
        # print >> sys.stderr, "Chunking trees..."
        chunk_tree = self._chunker.chunk(sent)
        # print >> sys.stderr, "Trees chunked..."
        # print >> sys.stderr, "Parsing..."
        tree = self._parser.parse(chunk_tree, catgetter=catgetter)
        # print >> sys.stderr, "Parsed..."
        # print >> sys.stderr, "Flattenning..."
        self._flatten(tree, ('AC', 'NC', 'FVG', 'IVG'))
        # print >> sys.stderr, "Flattenned..."
        # tree.pretty_print(feat_print=print_feat)
        return tree

    def _prepare_tokens(self, sent):
        for token in sent:
            verb_type = data.finite_verbs.get(token['form'], default=None)
            if verb_type is not None:
                token['pos'] = 'V{0}FIN'.format(verb_type)

    def _flatten(self, node, labels, parent=None):
        if not isinstance(node, Tree):
            return
        for child in node:
            self._flatten(child, labels, parent=node)
        if node.label in labels:
            parent.replace(node, *node)

    def _setup_parser(self):
        define = self._parser.define
        add_rule = self._parser.add_rule

        ##########
        # Macros #
        ##########

        define('VFIN',
            '''
            <VAFIN>
            <VMFIN>
            <VVFIN>
            ''')

        define('VINF',
            '''
            <VAINF>
            <VAPP>
            <VMINF>
            <VMPP>
            <VVINF>
            <VVIZU>
            <VVPP>
            ''')

        define('V',
            '''
            <VAFIN>
            <VMFIN>
            <VVFIN>
            <VAINF>
            <VAPP>
            <VMINF>
            <VMPP>
            <VVINF>
            <VVIZU>
            <VVPP>
            ''')

        define('PUNCT',
            '''
            <$,>
            <$(>
            <$.>
            ''')

        define('DASH',
           '''
           <EM_DASH>
           <EN_DASH>
           <FIGURE_DASH>
           <HORIZONTAL_BAR>
           ''')

        define('VG',
            '''
            <FVG>
            <IVG>
            ''')

        define('CLAUSE',
            '''
            <RelCl>
            <InfCl>
            <IntCl>
            <SntSubCl>
            <InfSubCl>
            <MainCl>
            ''')

        define('BASIC_CONTENT',
            '''
            (?:
                [^%PUNCT%<KON>%VG%]+
                (?:
                    <KON>?
                    [^%PUNCT%<KON>%VG%]+
                )?
            )*
            ''')

        define('CONTENT',
            '''
            (?:
                [^%PUNCT%<KON>%VG%]
                (?:
                    [<$,><KON>]?
                    [^%PUNCT%<KON>%VG%]
                |
                    [%CLAUSE%]
                    <$,>
                )?
            )*
            ''')

        define('BASIC_TRAILER',
            '''
            (?:
                [<APPR><APPRART><KOKOM>]
                [^%PUNCT%<KON>%VG%]+
            )
            ''')

        ##########################
        # Parenthesized segments #
        ##########################

        for ldelim, rdelim in data.DELIMS.iteritems():
            ldelim_name = data.DELIM_NAMES[ldelim]
            rdelim_name = data.DELIM_NAMES[rdelim]
            add_rule('Paren', '<{0}>[^<{1}>]+<{1}>'.format(ldelim_name,
                                                           rdelim_name))

        ###############
        # Verb groups #
        ###############

        def get_verb(match, group=0):
            main, modal, aux = None, None, None
            for token in match[group]:
                pos = token['pos']
                if pos.startswith('VV'):
                    main = token
                elif pos.startswith('VM'):
                    modal = token
                elif pos.startswith('VA'):
                    aux = token
            if main:
                return main
            elif modal:
                return modal
            return aux

        add_rule('FVG',
            '''
            (
                <PTKZU>?
                [%VINF%]+
                [%VFIN%]
            |
                <VVINF>        # gehen !!! added by W. Sidorenko (remove if it causes errors)
                <VVINF>        # lassen
            )
            (?:
                [^<NC><PC>]
            |
                $
            )
            ''',
            group=1,
            feats=lambda match: {'verb': get_verb(match, group=1)},
            level=5)

        add_rule('FVG',
            '''
            [%VFIN%]
            [%VINF%]*
            ''',
            feats=lambda match: {'verb': get_verb(match)},
            level=5)

        add_rule('IVG',
            '''
            [%VINF%]*
            <PTKZU>?
            [%VINF%]+
            ''',
            feats=lambda match: {'verb': get_verb(match)},
            level=5)

        ################################
        # Basic clauses (no embedding) #
        ################################

        add_rule('RelCl',
            '''
            <$,>                    # comma
            <APPR>?                 # optional preposition
            [<PRELS><PRELAT>]       # relative pronoun
            %BASIC_CONTENT%         # clause content
            (
                [%VG%]              # verb group (error tolerance: should actually be finite)
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=6)

        add_rule('RelCl',
            '''
            <KON>                   # conjunction
            (
                <APPR>?             # optional preposition
                [<PRELS><PRELAT>]   # relative pronoun
                %BASIC_CONTENT%     # clause content
                (
                    [%VG%]          # verb group (error tolerance: should actually be finite)
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=7)

        add_rule('RelCl',
            '''
            <RelCl>                 # relative clause
            <KON>                   # conjunction
            (
                %BASIC_CONTENT%     # clause content
                (
                    [%VG%]          # verb group (error tolerance: should actually be finite)
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=7)

        def complex_that(match):
            tokens = list(match[1][0].iter_terminals())
            return (match_(tokens, ('Dadurch', 'Dafür', 'Dafuer')) or
                    match_(tokens, 'Aufgrund', 'dessen') or
                    match_(tokens, 'Auf', 'Grund', 'dessen'))

        add_rule('SntSubCl',
            '''
            ^                       # start of sentence
            (                       # cases like
                <AC>                # "Dadurch, dass"
            |
                <PC>                # or "Aufgrund dessen", dass
            )
            <$,>                    # comma
            <KOUS>                  # subordinating conjunction
            %BASIC_CONTENT%         # clause content
            (
                <FVG>               # finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            constraint=complex_that,
            level=7)

        add_rule('SntSubCl',
            '''
            (?:
                ^                   # start of sentence
            |
                <$,>                # or comma
            |
                <$(>                # or dash
            )
            [<AC><APPR>]?           # optional adverb or preposition ("außer wenn ...")
            <KOUS>                  # subordinating conjunction
            %BASIC_CONTENT%         # clause content
            (
                <FVG>               # finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=7)

        add_rule('SntSubCl',
            '''
            <KON>                   # conjunction
            (
                <APPR>?             # optional preposition ("außer wenn ...")
                <KOUS>
                %BASIC_CONTENT%     # clause content
                (
                    <FVG>           # finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        add_rule('SntSubCl',
            '''
            <SntSubCl>              # sentential subordinate clause
            <KON>                   # conjunction
            (
                %BASIC_CONTENT%     # clause content
                (
                    FVG>            # finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        add_rule('InfSubCl',
            '''
            (?:
                ^                   # start of sentence
            |
                <$,>                # or comma
            )
            [<AC><APPR>]?           # optional adverb or preposition ("außer um ...")
            <KOUI>                  # subordinating conjunction
            %BASIC_CONTENT%         # clause content
            (
                <IVG>               # non-finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=7)

        add_rule('InfSubCl',
            '''
            <KON>                   # conjunction
            (
                <KOUI>              # subordinating conjunction
                %BASIC_CONTENT%     # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        add_rule('InfSubCl',
            '''
            <InfSubCl>              # non-finite subordinate clause
            <KON>                   # conjunction
            (
                %BASIC_CONTENT%     # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        add_rule('IntCl',
            '''
            (?:
                ^                   # start of sentence
            |
                <$,>                # or comma
            |
                <KON>                # or conjunction
            )
            [<PWS><PWAT><PWAV>]     # interrogative pronoun
            %BASIC_CONTENT%         # clause content
            (
                [%VG%]              # verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=7)

        add_rule('IntCl',
            '''
            <KON>
            (
                [<PWS><PWAT><PWAV>] # interrogative pronoun
                %BASIC_CONTENT%     # clause content
                (
                    [%VG%]          # verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        add_rule('IntCl',
            '''
            <IntCl>
            <KON>
            (
                %BASIC_CONTENT%     # clause content
                (
                    [%VG%]          # verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        ######################
        # Clause combination #
        ######################

        add_rule('IntCl',
            '''
            <IntCl>
            <RelCl>
            ''',
            level=9)

        add_rule('IntCl',
            '''
            <IntCl>
            (?:
                <KON>?
                <IntCl>
            )+
            ''',
            level=10)

        add_rule('RelCl',
            '''
            <RelCl>
            <SntSubCl>
            ''',
            level=10)

        add_rule('RelCl',
            '''
            <RelCl>
            <InfSubCl>
            ''',
            level=10)

        add_rule('RelCl',
            '''
            <RelCl>
            <Paren>
            ''',
            level=10)

        add_rule('RelCl',
            '''
            <RelCl>
            (?:
                <KON>?
                <RelCl>
            )+
            ''',
            level=11)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <RelCl>
            ''',
            level=11)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <IntCl>
            ''',
            level=11)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <InfSubCl>
            ''',
            level=11)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            (?:
                <KON>
                <SntSubCl>
            )+
            ''',
            level=11)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            <RelCl>
            ''',
            level=11)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            <IntCl>
            ''',
            level=11)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            (?:
                <KON>
                <InfSubCl>
            )+
            ''',
            level=11)

        ######################################
        # Basic clauses (embedding possible) #
        ######################################

        add_rule('RelCl',
            '''
            <$,>                    # comma
            <APPR>?                 # optional preposition
            [<PRELS><PRELAT>]       # relative pronoun
            %CONTENT%               # clause content
            (
                [%VG%]              # verb group (error tolerance: should actually be finite)
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=12)

        add_rule('RelCl',
            '''
            <KON>                   # conjunction
            (
                <APPR>?             # optional preposition
                [<PRELS><PRELAT>]   # relative pronoun
                %CONTENT%           # clause content
                (
                    [%VG%]          # verb group (error tolerance: should actually be finite)
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('RelCl',
            '''
            <RelCl>                 # relative clause
            <KON>                   # conjunction
            (
                %CONTENT%           # clause content
                (
                    [%VG%]          # verb group (error tolerance: should actually be finite)
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('SntSubCl',
            '''
            ^                       # start of sentence
            (                       # cases like
                <AC>                # "Dadurch, dass"
            |
                <PC>                # or "Aufgrund dessen", dass
            )
            <$,>                    # comma
            <KOUS>                  # subordinating conjunction
            %BASIC_CONTENT%         # clause content
            (
                <FVG>               # finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            constraint=complex_that,
            level=12)

        add_rule('SntSubCl',
            '''
            (?:
                ^                   # start of sentence
            |
                <$,>                # or comma
            )
            [<AC><APPR>]?           # optional adverb or preposition ("außer wenn ...")
            <KOUS>                  # subordinating conjunction
            %CONTENT%               # clause content
            (
                <FVG>               # finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=12)

        add_rule('SntSubCl',
            '''
            <KON>                   # conjunction
            (
                [<AC><APPR>]?       # optional adverb or preposition ("außer wenn ...")
                <KOUS>              # subordinating conjunction
                %CONTENT%           # clause content
                (
                    <FVG>           # finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('SntSubCl',
            '''
            <SntSubCl>              # sentential subordinate clause
            <KON>                   # conjunction
            (
                %CONTENT%           # clause content
                (
                    <FVG>           # finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('InfSubCl',
            '''
            (?:
                ^                   # start of sentence
            |
                <$,>                # or comma
            )
            [<AC><APPR>]?           # optional adverb or preposition ("außer um ...")
            <KOUI>                  # subordinating conjunction
            %CONTENT%               # clause content
            (
                <IVG>               # non-finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=12)

        add_rule('InfSubCl',
            '''
            <KON>                   # conjunction
            (
                [<AC><APPR>]?       # optional adverb or preposition ("außer um ...")
                <KOUI>              # subordinating conjunction
                %CONTENT%           # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('InfSubCl',
            '''
            <InfSubCl>              # non-finite subordinate clause
            <KON>                   # conjunction
            (
                %CONTENT%           # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('IntCl',
            '''
            (?:
                ^                   # start of sentence
            |
                <$,>                # or comma
            )
            [<PWS><PWAT><PWAV>]     # interrogative pronoun
            %CONTENT%               # clause content
            (
                [%VG%]              # verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=12)

        add_rule('IntCl',
            '''
            <KON>
            (
                [<PWS><PWAT><PWAV>] # interrogative pronoun
                %CONTENT%           # clause content
                (
                    [%VG%]          # verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('IntCl',
            '''
            <IntCl>
            <KON>
            (
                %CONTENT%           # clause content
                (
                    [%VG%]          # verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('InfCl',
            '''
            (?:
                ^                   # start of a sentence
            |
                <$,>                # or comma
            |
                <KON>                # or conjunction
            )
            (?:
                [^<PRELS><PRELAT><KOUS><KOUI><FVG>]
                %CONTENT%
                <NC>
                %CONTENT%
            )?
            (
                <IVG>               # non-finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            <$.>?                   # optional end of sentence punctuation
            ''',
                 group = 1,
                 feats=lambda match: {'verb': match[1][0].get('verb')},
                 level=12)

        add_rule('InfCl',
            '''
            <InfCl>                 # non-finite (complement) clause
            <KON>                   # conjunction
            (
                %CONTENT%           # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                <$.>?               # optional end of sentence punctuation
            )
            ''',
                 group = 1,
                 feats = lambda match: {'verb': match[2][0].get('verb')},
                 level = 13)

        ######################
        # Clause combination #
        ######################

        add_rule('InfCl',
            '''
            <InfCl>
            <RelCl>
            ''',
            level=14)

        add_rule('InfCl',
            '''
            <InfCl>
            (?:
                <KON>?
                <InfCl>
            )+
           ''',
            level=15)

        add_rule('IntCl',
            '''
            <IntCl>
            <RelCl>
            ''',
            level=14)

        add_rule('IntCl',
            '''
            <IntCl>
            (?:
                <KON>?
                <IntCl>
            )+
            ''',
            level=15)

        add_rule('RelCl',
            '''
            <RelCl>
            <SntSubCl>
            ''',
            level=15)

        add_rule('RelCl',
            '''
            <RelCl>
            <InfSubCl>
            ''',
            level=15)

        add_rule('RelCl',
            '''
            <RelCl>
            <InfCl>
            ''',
            level=15)

        add_rule('RelCl',
            '''
            <RelCl>
            (?:
                <KON>?
                <RelCl>
            )+
            ''',
            level=16)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <RelCl>
            ''',
            level=16)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <IntCl>
            ''',
            level=16)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <InfSubCl>
            ''',
            level=16)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            (?:
                <KON>
                <SntSubCl>
            )+
            ''',
            level=16)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            <RelCl>
            ''',
            level=16)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            <IntCl>
            ''',
            level=16)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            (?:
                <KON>
                <InfSubCl>
            )+
            ''',
            level=16)

        ################
        # Main clauses #
        ################

        # Verb-first or verb-second main clause

        def get_verb_feats(match):
            fin = match[1][0].get('verb')
            if fin is not None:
                if fin['pos'].startswith('VV'):
                    try:
                        particle = match[2][0]['lemma']
                    except IndexError:
                        particle = None
                    return {'verb': fin, 'verb_part': particle}
            try:
                inf = match[3][0].get('verb')
            except IndexError:
                return {'verb': fin}
            if inf is not None and inf['pos'].startswith(('VM', 'VV')):
                return {'verb': inf}
            return {'verb': fin}

        add_rule('MainCl',
            '''
            <$,>
            <KON>
            (
                <SntSubCl>
            )
            [%CLAUSE%]*
            <$.>?
            ''',
            feats=lambda match: match[1][0].feats,
            level=17)

        add_rule('MainCl',
            '''
            (
                <$,>
                <KON>
                (?:
                    [^<KON><MainCl><FVG>]+
                    (?:
                        <KON>?
                        [^<KON><MainCl><FVG>]+
                    )?
                )*
                (
                    <FVG>
                )
                [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                (?:
                    (?:
                        [%CLAUSE%]
                        (?:
                            <$,>
                            [%CLAUSE%]
                        )*
                    )?
                    [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                )*
                (
                    <PTKVZ>
                )?
                (
                    <IVG>
                )?
                %BASIC_TRAILER%?
                (?:
                    [%CLAUSE%]
                    (?:
                        <$,>
                        [%CLAUSE%]
                    )*
                )?
            )
            <$,>
            ''',
            group=1,
            level=17)

        add_rule('MainCl',
            '''
            (
                (?:
                    ^
                    <KON>
                )?
                (?:
                    [^<KON><MainCl><IVG>]
                    (?:
                        <KON>
                        [^<KON><MainCl><IVG>]
                    )?
                )*
                (
                    <IVG>
                )
                [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                (?:
                    [%CLAUSE%]
                    <$,>
                    [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                )*
                (
                    <PTKVZ>
                )?
            )
            [%PUNCT%]
            <AC>
            ''', group=1, level=17)

        add_rule('MainCl',
            '''
            (?:
                ^
                <KON>
            )?
            (?:
                [^<KON><MainCl><FVG>]
                (?:
                    <KON>
                    [^<KON><MainCl><FVG>]
                )?
            )*
            (
                <FVG>
            )
            [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
            (?:
                [%CLAUSE%]
                <$,>?
            )*
            [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
            (
                <PTKVZ>
            )?
            (
                <IVG>                                       # either a non-finite verb group
            |                                               # or (error tolerance)
                <FVG>                                       # a finite verb group if
                [%PUNCT%]                                   # immediately followed by punctuation
            )?
            (?:
                [%DASH%%PUNCT%]
                (?:
                    [^%VG%%DASH%%PUNCT%]+
                    (?:
                        $
                    |
                        [%DASH%%PUNCT%]
                    )
                )?
            |
                [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%]*
                (?:
                    [%CLAUSE%]
                    <$,>?
                )*
            )?
            <$.>?
            ''',
                 feats=get_verb_feats,
                 level=18)

        add_rule('MainCl',
            '''
            (
                <MainCl>
            )
            (?:
                [<KON><$,>]
                <MainCl>
            )+
            ''',
            feats=lambda match: match[1][0].feats,
            level=19)

        add_rule('MainCl',
            '''
            ^
            (?:
            <NE>+
            (?: <$,> | <$(> )
            )?
            (
                <SntSubCl>
            )
            [%CLAUSE%]+
            <$.>?
            ''',
            feats=lambda match: match[1][0].feats,
            level=17)

        # Catch-all rule (fallback).

        add_rule('ANY',
            '''
            [^<MainCl>]+
            ''',
            level=20)
