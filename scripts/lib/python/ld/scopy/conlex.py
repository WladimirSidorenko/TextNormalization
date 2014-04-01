#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from functools import partial
import os

from lxml import etree

data = partial(os.path.join, os.path.dirname(__file__), 'data')


class ConLex(object):
    '''Simple API for the con lexicon'''

    def __init__(self, path=None):
        self._lex = etree.parse(path or data('conlex.xml'))

    def xpath(self, expr):
        '''Evaluate the XPath expression ``expr`` against the lexicon.'''
        return self._lex.xpath(expr)

    def get(self, con, section=None):
        '''Return the entry of the con ``con``, or None if no such
        entry exists. If ``section`` is given, that part of the
        connective's entry that matches the xpath expression will be
        returned.
        '''
        entry = self._find_entry(con)
        if not entry:
            return
        elif section is None:
            return entry[0]
        return entry[0].xpath(section)

    def type(self, con):
        '''Return the type of the connective ``con``, which is either
        adv(erbial), conj(unction), subj(unction), prep(ositional) or
        None, if no entry exists for it.
        '''
        cat = self.get(con, 'syn/@type')
        if not cat:
            return None
        return cat[0]

    def maybe_anteponed(self, con):
        '''Return true if the connective ``con`` may be
        anteponed, false otherwise.
        '''
        entry = self._find_entry(con)
        if entry is None:
            return False
        return True if entry[0].xpath('syn/order/ante[text()="1"]') else False

    def maybe_postponed(self, con):
        '''Return true if the connective ``con`` may be
        postponed, false otherwise.
        '''
        entry = self._find_entry(con)
        if entry is None:
            return False
        return True if entry[0].xpath('syn/order/post[text()="1"]') else False

    def maybe_inserted(self, con):
        '''Return true if the connective ``con`` may be
        inserted, false otherwise.
        '''
        entry = self._find_entry(con)
        if entry is None:
            return False
        if entry[0].xpath('syn/order/insert[text()="1"]'):
            return True
        return False

    def relations(self, con):
        '''Return a list of relations for the connective ``con``.'''
        entry = self._find_entry(con)
        if entry is None:
            return []
        return entry[0].xpath('syn/rel/text()'.format(con))

    def has_relation(self, con, rel):
        '''Return true if the connective ``con`` can signal the
        relation ``rel``, false otherwise.
        '''
        entry = self._find_entry(con)
        if entry is None:
            return False
        elif entry[0].xpath('syn/rel[text()="{0}"]'.format(rel)):
            return True
        return False

    def _find_entry(self, con):
        if not isinstance(con, basestring):
            con = ' '.join(con).strip()
        return self._lex.xpath(
            'entry[orth[normalize-space(string())="{0}"]]'.format(con))

if __name__ == '__main__':
    conlex = ConLex()
    print conlex.xpath('count(entry/syn[@type="conj"])')
