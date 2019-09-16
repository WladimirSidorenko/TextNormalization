#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

try:
    from paula.directory_reader import PaulaProject
except ImportError:
    sys.stderr.write("Paula isn't installed\n")
    sys.stderr.write('Exiting ...')
    sys.exit(1)


class PaulaImporter(object):
    '''Class for the conversion of PAULA projects into EDSeg's generic
    input format.

    '''
    def __init__(self, project):
        self.project = project
        self._model = project.model
        layer_map = self._model.userTypeLayerMap
        self._lemma_feat = layer_map['Tree_Lemma'][0]
        self._pos_feat = layer_map['mate.syntax_POS'][0]
        self._feats_feat = layer_map['mate.syntax_FEATS'][0]
        self._head_feat = layer_map['mate.syntax_HEAD'][0]
        self._deprel_feat = layer_map['mate.syntax_HEAD_DEPREL'][0]

    @classmethod
    def from_directory(cls, path):
        '''Import a PAULA project from ``path``.'''
        project = PaulaProject()
        project.import_from_directory(path)
        return cls(project)

    def __iter__(self):
        '''Alias for ``PaulaImporter.iter_sentences()``.'''
        return self.iter_sentences()

    def iter_sentences(self):
        '''Return an iterator over sentences. Each sentence is
        represented as a list of tokens and each token as a dict.
        Tokens provide the following attributes:

            - ID,
            - wordform,
            - lemma,
            - part-of-speech,
            - morphological features,
            - the ID of the (dependency) head,
            - the dependency relation between the token and its head,
            - the token's PAULA ID

        For reasons of convenience, IDs are mapped onto list indices.
        '''
        paula_sents = self._model.userTypeLayerMap['sent'][0].items()
        paula_sents.sort(key=lambda item: item[1].pid)
        for pid, paula_sent in paula_sents:
            sent = []
            pid2tid_map = {'tok_0': -1}
            for tid, token in enumerate(paula_sent.ext):
                lemma = token.feature.get(self._lemma_feat)
                if lemma is None:
                    # Take the surface form if the lemma is unknown.
                    lemma = token.getString()
                else:
                    lemma = lemma.value
                sent.append({
                    'id': tid,
                    'pid': token.pid,
                    'form': token.getString(),
                    'lemma': lemma,
                    'pos': token.feature[self._pos_feat].value,
                    'feats': token.feature[self._feats_feat].value,
                    'head': token.feature[self._head_feat].value,
                    'deprel': token.feature[self._deprel_feat].value,
                    })
                pid2tid_map[token.pid] = tid
            for token in sent:
                token['head'] = pid2tid_map.get(token['head'], -1)
            yield sent
