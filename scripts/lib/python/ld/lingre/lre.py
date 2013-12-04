#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
import re
import sys

from .. import DEFAULT_RE, RuleFormatError, skip_comments
from .  import RE_OPTIONS, RE_OPTIONS_SEP
from lre_match import MultiMatch
from alt_fio import AltFileInput

##################################################################
# Constants
EXT_FLAGS = set(['re.WORDS'])

##################################################################
# Classes
class RegExpStruct(list):
    '''Container class for holding list of regexps with their options.'''
    def __init__(self):
        '''Instantiate a representative of RegExpStruct class.'''
        super(RegExpStruct, self).__init__([[], 0])


class RegExp():
    '''Class representing single regular expression with options.'''
    def __init__(self, flag_str, *regexps):
        '''Create an instance of RegExp.'''
        self.flags, self.ext_flags = self.__parse_flags(flag_str)
        self.re = self.__compile(*regexps)

    def __parse_flags(self, flag_str):
        '''Subdivide flags into natively supported and customarily added.'''
        res_flags = 0
        ext_flags = []
        # split flags string on separators and remove empty elements if any
        flag_str = RE_OPTIONS.sub(r"\1", flag_str) or flag_str
        flags = filter(None, RE_OPTIONS_SEP.split(flag_str))
        for flag in flags:
            try:
                res_flags |= eval(flag , {'__builtins__': None, 're': re})
            except AttributeError:
                self.__check_ext_flag(flag)
                ext_flags.append(flag)
        return res_flags, set(ext_flags)

    def __check_ext_flag(self, flag):
        '''Check if flag is in set of extended flags.'''
        if flag in EXT_FLAGS:
            return flag
        raise RuleFormatError(msg = "Option " + flag.encode('utf-8') + \
                                  " is not supported")

    def __compile(self, *regexps):
        '''Compile self.re according to self.flags and self.ext_flags.'''
        lbound = r'(?:'
        rbound = r')'
        if 're.WORDS' in self.ext_flags:
            lbound = r'\b' + lbound
            rbound += r'(?!\w|-)'
        return re.compile(lbound + '|'.join(regexps) + rbound, self.flags)


class MultiRegExp():
    '''Container class used to hold multiple compiled regexps.'''
    def __init__(self, ifile, encd = 'utf8', \
                     no_inner_groups = False, \
                     istring_hook = lambda istring: [istring]):
        '''Load regular expressions from text file.

        Read input file passed as argument and convert lines contained there to
        a RegExp union, i.e. regexps separated by | (OR). If istring_hook is
        supplied, it should be a function called for every input line except
        for lines with compiler directives. Return value of this function is a
        list.'''
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
            self.re_list = self.__load(ifile, encd = self.encd)


    def finditer(self, istring):
        '''Find all non-overlapping spans of text matching regexps from list.'''
        output = []
        groups = ()
        for re in self.re_list:
            # collect all possible matches as tuples for all regexps
            for match in re.finditer(istring):
                groups = match.groups()
                # iterate over all possible groups of re
                for gid in xrange(len(groups)):
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

    def compile(self, re_list):
        '''Transform list of regexps and options to a list of compiled
        regexps'''
        result_list = []
        if re_list[0][0]:
            # unite regexps into groups
            if self.no_inner_groups:
                lbracket = r'('
            else:
                lbracket = r'(?:'
            rbracket = r')'
            # adapt boundaries according to options
            comp_rexp = ''
            for (rexps, ropts) in re_list:
                comp_rexp = re.compile(lbracket + '|'.join(rexps) + rbracket, ropts)
                result_list.append(comp_rexp)
        return result_list

    def __load(self, ifile, encd):
        '''Load regular expressions from ifile. '''
        re_list = [RegExpStruct()]
        match = None
        cnt = 0
        finput = AltFileInput(ifile, encoding = encd)

        for line in finput:
            match = RE_OPTIONS.match(line)
            # different regexp options will separate different
            # chunks of regular expressions
            if match:
                # increment counter only if we have already seen any
                # regexps before
                if cnt != 0 or re_list[0][0]:
                    re_list.append(RegExpStruct())
                    cnt += 1
                # securily interpret options passed as strings as valid python
                # code
                re_list[cnt][1] = self.__parse_options(match)
            else:
                # strip off comments
                line = skip_comments(line)
                # and remember the line if it is not empty
                if line:
                    re_list[cnt][0].extend(self.istring_hook(line))
        return self.compile(re_list)

    def __parse_options(self, m_obj):
        '''Evaluate options for regular expresions.'''
        opts = RE_OPTIONS_SEP.split(m_obj.group(1))
        res_opts = 0
        for opt in opts:
            # evaluate options and collect them
            res_opts |= eval(opt, {'__builtins__': None, 're': re})
        return res_opts
