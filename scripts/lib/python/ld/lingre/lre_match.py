#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
import copy as __copy__

##################################################################
# Classes
class MultiMatch(list):
    '''Container class used to hold match objects produced by MultiRegExp.'''
    def __init__(self, matches=[]):
        '''Create an instance of MultiRegExpMatch.'''
        super(self.__class__, self).__init__(self.select_llongest(matches))

    def select_llongest(self, mcontainer):
        '''Select leftmost longest matches from multiple possible.'''
        result = []
        prev_start, prev_end = -1, -1
        match_el = None
        mcontainer.sort()       # mcontainer is supposed to be a list
                                # of match spans, i.e. tuples with
                                # start and end points of a match

        # we are running from the end of mcontainer due to
        # implementation of lists
        while mcontainer:
            match_el = mcontainer.pop()
            # don't care about empty matches
            if not match_el:
                continue
            start, end = match_el
            if end <= prev_start: # because end is 1 char more than
                                  # the regexp actually matched
                # we have come across a non-overlapping match
                result.append((prev_start, prev_end))
            elif ( start > prev_start > -1 ) and \
                    ( start == prev_start and end < prev_end ):
                # don't remember overlapping matches, which aren't
                # leftmost longest
                continue
            prev_start, prev_end = start, end
        if match_el:
            result.append((prev_start, prev_end))
        # restore original order of things
        result.reverse()
        return result


    def select_nonintersect(self, mcontainer2):
        '''Leave in container1 only elements not intersecting with container2.'''
        # it's assumed that both containers are sorted
        mcontainer1 = __copy__.copy(self)
        cnt1, stop1 = 0, len(mcontainer1)
        cnt2, stop2 = 0, len(mcontainer2)
        start1, end1 = None, None
        start2, end2 = None, None
        # run through both containers simultaneously and determine
        # on the fly if an element from container1 intersects with
        # an element from container2, delete the former if yes
        while cnt1 < stop1 and cnt2 < stop2:
            start1, end1 = mcontainer1[cnt1]
            start2, end2 = mcontainer2[cnt2]
            if start1 >= start2:
                if start1 <= end2:
                    # if there was an intersection of mcontainer1
                    # element with an element of mcontainer2 - delete
                    # the former
                    del mcontainer1[cnt1]
                    stop1 -= 1
                else:
                    cnt2 += 1
            else:
                cnt1 += 1
        return mcontainer1
