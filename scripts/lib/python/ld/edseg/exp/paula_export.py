#!/usr/bin/env python
# -*- coding: utf-8 -*-


class PaulaExporter(object):
    '''EDSeg PAULA (Potsdamer Austauschformat für linguistische
    Annotation) exporter
    '''
    EDSEG = 'edseg'
    SDS = 'sds'
    EDS = 'eds'
    EDS_FEATS = ('type', 'parent')

    def __init__(self, project, path=None):
        self.project = project
        self.path = path
        self._model = project.model
        self._tok_layer = self._model.getTokenLayer()[0]
        self._eds_layer = None
        self._eds_id = None
        self._sds_layer = None
        self._sds_id = None
        self._feat_layers = None

    def on_start(self):
        '''Create all segmentation-related markable and feature layers
        and reset SDS and EDS counters.
        '''
        self._eds_layer = self._model.createMarkLayer(
            '{0}.{1}'.format(self.EDSEG, self.EDS))
        self._eds_id = 0
        self._sds_layer = self._model.createMarkLayer(
            '{0}.{1}'.format(self.EDSEG, self.SDS))
        self._sds_id = 0
        create_feat = self._model.createFeatLayer
        self._feat_layers = dict((feat,
                                  create_feat('{0}.{1}_{2}'.format(self.EDSEG,
                                                                   self.EDS,
                                                                   feat)))
                                 for feat in self.EDS_FEATS)

    def on_finish(self):
        '''Write all segmentation-related layers and export the project
        to the given path.
        '''
        self._eds_layer.write()
        self._sds_layer.write()
        for layer in self._feat_layers.itervalues():
            layer.write()
        if self.path is not None:
            self.project.export(self.path)

    def on_sds(self, sent_no, sds):
        '''Add new SDS and EDS markables as well as the corresponding
        EDS features.
        '''
        sds_tokens = []
        for eds in sds.iter('EDS'):
            tokens = [self._tok_layer[tok['pid']]
                      for tok in list(eds.iter_terminals())]
            # if not tokens:
            #     continue
            self._add_eds(tokens, eds)
            sds_tokens.extend(tokens)
        self._add_sds(sds_tokens)

    def _add_eds(self, tokens, node):
        pid = '{0}_{1}'.format(self.EDS, self._eds_id)
        self._eds_layer.createMark(pid, tokens)
        mark = self._eds_layer[pid]
        for name, value in node.feats.iteritems():
            if name == 'embedded':
                name = 'parent'
                value = '{0}_{1}'.format(self.EDS, self._eds_id - 1)
            layer = self._feat_layers[name]
            layer.addEntry(mark, None, value)
        self._eds_id += 1

    def _add_sds(self, tokens):
        self._sds_layer.createMark('{0}_{1}'.format(self.SDS, self._sds_id),
                                                    tokens)
        self._sds_id += 1
