#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################
from scoper import ScopeFinder
import scopyfilter as filt

##################################################################
class DefaultScopeFinder(ScopeFinder):
    '''Pre-initialized version of Scopy. Uses the filters proposed in
    (Prasad et al, 2010).
    '''
    def __init__(self, **kwargs):
        ScopeFinder.__init__(self, **kwargs)
        # filt1 = filt.InOpaqueDirectSpeechFilter(self)
        # filt2 = filt.InOpaqueParenFilter(self)
        # filt3 = filt.OutsideOpaqueZoneFilter(self)
        # filt4 = filt.ContrastFilter(self)

        # # Chain filters
        # filt1.on_success(filt4)
        # filt1.on_failure(filt2)

        # filt2.on_success(filt4)
        # filt2.on_failure(filt3)

        # filt3.next(filt4)

        # self.filter_chain = filt1
