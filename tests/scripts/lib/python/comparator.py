#!/usr/bin/env python2.7

##################################################################
# Import Libraries
from ipopen import IPopen

##################################################################
# Class
class Comparator(IPopen):
    '''Processor used to make comparison between input and gold.

    This class provides only one public method - `cmp()', which
    is a wrapper around further comparison methods that you may
    specify. Additionally `cmp()' pre-processes compared data
    before passing them down for comparison.
    '''

    def __init__(self, func='_cmp', ignore_case=False):
        '''Create a Comparator instance for comparing 2 strings.'''
        self.ic = ignore_case
        # if user has specified a particular comparison method
        # from this class, that method will be used in cmp().
        self.cmp_func = getattr(self, func)

    def cmp(self, s1, s2):
        '''Public comparison method of the class.'''
        if self.ic:
            s1 = s1.lower()
            s2 = s2.lower()
        return self.cmp_func(s1, s2)

    def _cmp(self, s1, s2):
        '''Default comparison method used in this class.'''
        return cmp(s1, s2)
