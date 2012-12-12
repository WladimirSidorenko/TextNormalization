#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
from alt_fileinput import AltFileInput
from .. import __re__, DEFAULT_RE, skip_comments
from . import OPTIONS_RE
from .lre_match import MultiMatch

##################################################################
# Classes
class RegExpStruct(list):
    '''Container class for holding list of regexps with their options.'''
    def __init__(self):
        '''Instantiate a representative of RegExpStruct class.'''
        super(RegExpStruct, self).__init__([[], 0])


class MultiRegExp():
    '''Container class used to hold multiple compiled regexps.'''
    def __init__(self, ifile, encd = 'utf8', \
                     no_inner_groups = False, \
                     istring_hook = lambda istring: [istring]):
        '''Load regular expressions from text file.

        Read input file passed as argument and convert lines contained
        there to a RegExp union, i.e. regexps separated by | (OR). If
        istring_hook is supplied, it should be a function called for every
        input line except for lines with compiler directives, note that it
        should return a list.'''
        # re_list will hold multiple lists each consisting of 2
        # elements. The fist element will be a list of regular expressions
        # and the second one will be a list of flags.  E.g.
        # re_list = [[['papa', 'mama'], RE.UNICODE], [['sister'], RE.IGNORECASE]]
        self.encd = encd
        self.no_inner_groups = no_inner_groups
        self.istring_hook = istring_hook
        if not ifile:
            self.re_list = [DEFAULT_RE]
        else:
            self.__load(ifile, encd = self.encd)

    # overwriting default class
    def finditer(self, istring):
        '''Expanding default finditer operator.'''
        output = []
        groups = ()
        for re in self.re_list:
            # collect all possible matches as tuples for all regexps
            for match in re.finditer(istring):
                groups = match.groups()
                # iterate over all possible groups of re
                for gid in range(len(groups)):
                    # if a group wasn't empty, add its span to output
                    if groups[gid]:
                        # because match.groups() and match.group()
                        # differ by element 0
                        output.append(match.span(gid + 1))
                        # a single re is assumed to produce only one
                        # non-empty group
                        break
        # After all regexps matched, leave only valid,
        # non-overlapping, leftmost-longest spans
        return MultiMatch(output)


    def __load(self, ifile, encd):
        '''Load regular expressions from ifile. '''
        self.re_list = [RegExpStruct()]
        match = None
        cnt = 0
        finput = AltFileInput(ifile, encd = encd)

        for line in finput:
            match = OPTIONS_RE.search(line)
            # different regexp options will separate different
            # chunks of regular expressions
            if match:
                # increment counter only if we have already seen any
                # regexps before
                if cnt != 0 or self.re_list[0][0]:
                    self.re_list.append(RegExpStruct())
                    cnt += 1
                # securily interpret options passed as strings as
                # valid python code (temporarily inveiling the true
                # nature or __re__)
                self.re_list[cnt][1] = eval(match.group(1), \
                                                {'__builtins__': None}, \
                                                {'re': __re__})
            else:
                # strip off comments
                line = skip_comments(line)
                # and remember the line if it is not empty
                if line:
                    self.re_list[cnt][0].extend(self.istring_hook(line))

        if self.re_list[0][0]:
            # unite regexps into groups
            lbracket = r'(' if self.no_inner_groups else r'(?:'
            self.re_list = [__re__.compile(lbracket + '|'.join(rexps) + r')', ropts) \
                                for (rexps, ropts) in self.re_list]

