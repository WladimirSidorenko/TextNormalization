#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xml.dom.minidom import Document


class XMLExporter(object):
    '''EDSeg XML exporter'''

    ROOT_TAG = 'discourse'
    SDS_TAG = 'sds'
    EDS_TAG = 'eds'

    def __init__(self):
        self.root = None

    def on_start(self):
        '''Create the document root and initialize SDS and EDS
        counters.
        '''
        self._doc = Document()
        self.root = self.make_root()
        self._doc.appendChild(self.root)
        self._sds_buff = []
        self._sds_count = 0
        self._eds_count = 0

    def on_finish(self):
        '''Sort the SDS elements to reflect document order and add all
        SDS element to the root.
        '''
        self._sds_buff.sort()
        for sds_no, sds in self._sds_buff:
            self.root.appendChild(sds)

    def on_sds(self, sent_no, sds):
        '''Create new SDS and EDS elements.'''
        sds_elem = self.make_sds(sds)
        for eds in sds:
            sds_elem.appendChild(self._traverse(eds))
        self._sds_buff.append((sent_no, sds_elem))

    def as_string(self, indent='  ', encoding='utf-8'):
        '''Return a string representation of the XML.'''
        return self._doc.toprettyxml(indent=indent, encoding=encoding)

    def write_to(self, path_or_file):
        '''Write the XML to a given file path or file handle.'''
        if hasattr(path_or_file, 'write'):
            fp = path_or_file
        else:
            fp = open(path_or_file, 'wb')
        fp.write(self.as_string())
        if fp is not path_or_file:
            fp.close()
        return self

    def make_root(self):
        '''Create the document root (element). This method can be
        overwritten to customize the root element.
        '''
        return self._doc.createElement(self.ROOT_TAG)

    def make_sds(self, sds):
        '''Create an SDS element. This method can be overwritten to
        customize SDS elements.
        '''
        elem = self._doc.createElement(self.SDS_TAG)
        elem.setAttribute('id', 's{0}'.format(self.new_sds_id()))
        return elem

    def make_eds(self, eds):
        '''Create an EDS element. This method can be overwritten to
        customize EDS elements.
        '''
        elem = self._doc.createElement(self.EDS_TAG)
        elem.setAttribute('id', 'e{0}'.format(self.new_eds_id()))
        elem.setAttribute('type', eds.get('type'))
        return elem

    def make_token(self, tok):
        '''Create a token element. This method can be overwritten to
        customize token elements.
        '''
        return self._doc.createTextNode(tok['form'])

    def new_sds_id(self):
        '''Return the next free SDS ID.'''
        sds_id = self._sds_count
        self._sds_count += 1
        return sds_id

    def new_eds_id(self):
        '''Return the next free EDS ID.'''
        eds_id = self._eds_count
        self._eds_count += 1
        return eds_id

    def _traverse(self, node):
        if not hasattr(node, 'label'):
            return self.make_token(node)
        elem = self.make_eds(node)
        for child in node:
            elem.appendChild(self._traverse(child))
        return elem
