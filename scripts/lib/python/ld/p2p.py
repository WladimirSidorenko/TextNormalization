#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
import re
import sys

from alt_fio import AltFileInput
from stringtools import upcase_capitalize
from . import skip_comments, RuleFormatError
from lingre import RE_OPTIONS
from lingre.lre import RegExp

##################################################################
# Constants
RULE_SEPARATOR = re.compile(r'\s+-->\s+')
REPL_SEPARATOR = re.compile(r'\s*;;\s*')
STRING_REPL    = re.compile(r'^"(?:[^"]|\\")+"$')
REPL_FLAG      = 'REPLACED'

##################################################################
# Class
class P2P:
    '''Class for regexp-based transformation of input text.'''

    def __init__(self, file_name):
        '''Read P2P rules from file and populate instance.'''
        self.rules = []
        self.flags = ''
        ifile = AltFileInput(file_name)
        for line in ifile:
            self.__parse(line)

    def sub(self, iline, remember = False):
        '''Substitute substrings of iline according to self.rules.'''
        # Prepare a container for storing replaced fragments
        memory = []
        # Find all leftmost longest non-overlapping matches of
        # condition rules in input string
        instructions = self.__search(iline)
        if not instructions:
            return iline, memory
        # Sort all found substrings by their starting positions.  Instructions
        # at this moment have the form:
        # [((0, 3), <_sre.SRE_Match object at 0x7f7ff67a9990>, \
            # <function <lambda> at 0x7f7ff67a98c0>)]
        instructions.sort(key = lambda instr: instr[0])
        # prepare variables which later will be used for replacement
        match = repl_func = None
        line_start = repl_start = repl_end = _id = 0
        output = orig = replaced = ''
        # iterate over all matched pieces and apply corresponding change rules
        # to them
        for instr in instructions:
            (repl_start, repl_end), match_obj, repl_func = instr
            # append to output substring before matched part
            output += iline[line_start:repl_start]
            # get the part to be substituted
            orig = iline[repl_start:repl_end]
            # wrap replacement procedure into a safety clause due to its
            # potential danger
            try:
                replaced = upcase_capitalize(repl_func(match_obj), orig)
            except:
                print >> sys.stderr, "Failed to apply rule to:", orig, iline
                replaced = orig
            if remember and replaced != orig:
                # elements of `replaced' will have the form:
                # (start_of_replacement, length_of_replacement, \
                    # replacement_checksum, original_string)
                memory.append((REPL_FLAG, str(len(output)), str(len(replaced)), \
                                  replaced, orig))
            try:
                output += replaced
            except UnicodeDecodeError:
                print >> sys.stderr, "Output is", repr(output)
                print >> sys.stderr, "Replaced is", repr(replaced)
                raise
            # for the next iteration assume that the string begins at
            # the end position of current match
            line_start = repl_end
        output += iline[line_start:]
        return output, memory

    # make an alias for sub
    replace = sub

    def __search(self, iline):
        '''Search for all occurrences of all rule conditions in iline.

        Return value is a list of 3-tuples of form

        (rule_id, match_obj, ((start_group1, end_group1), \
        (start_group2, end_group2)))

        where the first value is the id of rule which produced this
        match, the second value is the match object, and the 3-rd
        value is a tuple of spans for each group contained in match.'''
        # matches is a temporary storage for information about text spans
        # captured by match, along with corresponding match objects and id-s of
        # rules, which produced match
        matches    = []
        groups_cnt = 0
        # iterate over all rules in this p2p
        for r_id, rule in enumerate(self.rules):
            # for range(), the upper boundary of the number of groups
            # should be one more than the real number of groups in
            # expression, since numeration begins with 1
            groups_cnt = rule.condition.groups + 1
            # iterate over all possible matches of condition of given rule
            for match in rule.condition.finditer(iline):
                # add to matches a 3-tuple, in which the 1-st element
                # will be a list of spans produced by all groups in
                # match, the 2-nd element - the reference to match
                # object itself, and the 3-rd element - the id of the
                # rule, which produced match.
                matches.append(MatchTuple(r_id, match, groups_cnt))
        # leave only matches which don't contradict with other matches
        # according to the leftmost-longest principle
        return self.__process_matches(matches)

    def __process_matches(self, matches):
        '''Clean overlapping matches according to llongest principle.

        For information about return value, see documentation of self.__search'''
        # skip emty matches
        if not matches:
            return matches
        # sort matches according to starting position of their 1-st
        # group and ending position of last group they capture
        matches.sort(key = lambda match_tuple: (match_tuple.gstart, match_tuple.gend))
        # intialize variables
        result = []
        mt = None
        m_end = cmp_status = 0
        mlength = len(matches)
        # TODO: simplify references to array elements:
        # matches[i], matches[j] - are additional calls, which should be omitted
        # iterate over all match_tuples
        for i in range(mlength):
            mt = matches[i]
            if not matches[i]:
                continue
            mt_end = mt.gend
            # check all possible next matches if they intersect with
            # the current one
            for j in range(i+1, mlength):
                if not matches[j]:
                    continue
                # if next match doesn't intersect with current one at all,
                # quit the loop
                if matches[j].gstart >= mt_end:
                    break
                cmp_status = self.__cmp_match(matches, i, j)
                # if current match didn't win in the leftmost-longest
                # competition, quit the loop, the element itself will be
                # deleted in self.__cmp_match()
                if cmp_status < 0:
                    break
            # append match if it survived
            if matches[i]:
                result += self.__match2rule(matches[i])
        return result

    def __cmp_match(self, match_container, idx1, idx2):
        '''Compare 2 matches and delete one of them.'''
        # check if 2 match tuples don't intersect in their groups
        match_t1, match_t2 = match_container[idx1], match_container[idx2]
        if self.__disjoint_match(match_t1.spans, match_t2.spans):
            # return 0 if yes, which means that both groups don't
            # intersect with each other
            return 0
        else:
            # Decide which of the 2 match tuples has higher
            # precedence.  The comparison criteria are considered in
            # following order:
            # - starting position of the first group,
            # - ending position of the last group,
            # - starting position of the whole regexp,
            # - ending position of the whole regexp,
            # - number of characters captured by all groups,
            # - number of capturing groups,
            # - order of how rules appeared in file.
            # Note, that starting positions are compared in reverse
            # order, since a rule starting earlier has a higher weight.
            cmp_status  = cmp(match_t2.gstart, match_t1.gstart) or \
                cmp(match_t1.gend, match_t2.gend) or \
                cmp(match_t2.start, match_t1.start) or \
                cmp(match_t1.end, match_t2.end) or \
                cmp(match_t1.captured_chars, match_t2.captured_chars) or \
                cmp(match_t1.captured_groups, match_t2.captured_groups) or \
                cmp(match_t2.rule_id, match_t1.rule_id)
            if cmp_status > 0:  # if match_t1 is the leading one, delete the
                                # second disturbing match tuple
                match_container[idx2] = None
                return 1
            else:
                # otherwise, delete the first match tuple
                match_container[idx1] = None
                return -1

    def __disjoint_match(self, spans1, spans2):
        '''Check if 2 match spans don't intersect with each other.'''
        i = j = cmp_status = 0
        while i < len(spans1) and j < len(spans2):
            s1, e1 = spans1[i]
            s2, e2 = spans2[j]
            # If s1 or s2 are less than 0, it means, that these groups
            # didn't contribute to match.
            # If the 1-st group ended before start of the 2-nd.
            if s1 < 0 or e1 <= s2:
                # proceed to next group from 1-st list
                i += 1
            elif s2 < 0 or e2 <= s1:
                j += 1
            else:
                return False
        return True

    def __match2rule(self, match_tuple):
        '''Return a list of 2-tuples with captured span and rule to apply.'''
        result = []
        match_obj   = match_tuple.match
        replacement = self.rules[match_tuple.rule_id].replacement.groups
        for i, span in enumerate(match_tuple.spans):
            if span[0] != -1:
                result.append(tuple([span, match_obj, replacement[i]]))
        # print >> __sys__.stderr, 'Result of match --> rule conversion:', result
        return result

    def __sub(self, replacement, match):
        '''Return string replaced with match.'''
        _id = SPEC_START + str(match.start()) + SPEC_SEP + \
            str(match.end() - match.start()) + SPEC_END
        return m_object.expand(replacement).format(id = _id)

    def __parse(self, iline):
        '''Parse input lines of P2P file.'''
        if RE_OPTIONS.match(iline):
            self.flags = iline
        iline = skip_comments(iline)
        if not iline:
            return None
        self.rules.append(Rule(iline, self.flags))

#################################
class Rule:
    '''Class providing correspondence between condition and replacement.'''
    def __init__(self, iline, flags = ''):
        '''TODO'''
        rule = RULE_SEPARATOR.split(iline)
        if len(rule) != 2:
            raise RuleFormatError('Missing rule separator in line: ' + iline)
        self.condition   = Condition(rule[0], flags)
        self.replacement = Replacement(rule[1])
        self.__check_consistency(iline)

    def __check_consistency(self, iline):
        '''Check that internal state of object is correct.'''
        lcond = self.condition.groups
        lrepl = len(self.replacement.groups)
        if not lcond or lcond != lrepl:
            raise RuleFormatError('''
Invalid # of groups in rule:
{:s}
{:d} groups in condition
{:d} groups in replacement'''.format(iline, lcond, lrepl))
        return True

#################################
class Condition:
    '''Matching part of P2P rule.'''

    def __init__(self, istring, flags):
        '''Create an instance of P2P condition.'''
        self.re     = RegExp(flags, istring).re
        self.groups = self.re.groups

    def finditer(self, iline):
        '''Return all possible matches of condition of given rule.'''
        return self.re.finditer(iline)

#################################
class Replacement:

    ''' '''

    def __init__(self, istring):
        '''Create an instance of p2p.Replacement.'''
        if istring[-2:] != ";;":
            raise RuleFormatError('''
Incorrect replacement format. Replacement should end with {:s}.
'''.format(REPL_SEPARATOR.pattern))
        self.groups = self.__parse(REPL_SEPARATOR.split(istring)[:-1])

    def __parse(self, repls):
        "Create replacement instructions for a single group."
        # if irule is empty return address of self.__empty
        return map(self.__parse_single, repls)

    def __parse_single(self, irule):
        '''Parse single replacement instruction and return a function.'''
        # if irule is simply a string with no additional methods called on it,
        # return an address to a method which will simply expand and format
        # this string appropriate to locale context at call time
        if STRING_REPL.match(irule):
            procedure =  lambda match, **local_vars: \
                    match.expand(irule[1:-1].format(**local_vars))
            return procedure
        # otherwise, assume it to be executable python code
        else:
            return lambda match, **local_vars: \
                eval(match.expand(irule), {'match': match, '__builtins__': None}, \
                         local_vars)

    def __empty(self, *args, **kwargs):
        '''Return empty string whenever called.'''
        return ""

#################################
class MatchTuple:
    '''Class for holding relevant information about condition matches.'''

    def __init__(self, rule_id, match_obj, groups_cnt):
        '''Create an instance of MatchTuple.'''
        self.rule_id  = rule_id
        self.match = match_obj
        self.spans = tuple([match_obj.span(i) for i in range(1, groups_cnt)])
        self.captured_chars  = self.__count_chars()
        self.captured_groups = len(self.spans)
        # start and end of whole regexp
        self.start = self.match.start()
        self.end   = self.match.end()
        # start of first group and end of last one
        self.gstart = self.spans[0][0]
        self.gend   = self.spans[-1][-1]

    def __repr__(self):
        '''String representation of object.'''
        result = '<'
        result += self.__class__.__name__ + ', '
        result += 'id=' + str(hex(id(self))) + ', '
        result += 'rule_id=' + str(self.rule_id) + ', '
        result += 'match=' + repr(self.match) + ', '
        result += 'spans=' + repr(self.spans)
        result += '>'
        return result

    def __count_chars(self):
        '''Count total number of characters captured by all groups.'''
        result = 0
        for start, end in self.spans:
            result += end - start
        return result
