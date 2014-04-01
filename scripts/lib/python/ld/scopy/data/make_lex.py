#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from lxml import etree


def make_lex(dimlex_path):
    dimlex = etree.parse(dimlex_path)
    with open(os.path.join(os.path.dirname(__file__), 'make_lex.xsl')) as fp:
        stylesheet = fp.read()
    transform = etree.XSLT(etree.XML(stylesheet))
    return transform(dimlex)


if __name__ == '__main__':
    try:
        dimlex_path = sys.argv[1]
    except IndexError:
        sys.exit(1)
    print make_lex(dimlex_path)
