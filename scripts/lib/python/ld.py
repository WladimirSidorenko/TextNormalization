##################################################################
# Importing Libraries
import re
import locale
import copy
import sys

##################################################################
# Defining Module Attributes

# module constants
OPTIONS_RE = re.compile('^##!\s*RE_OPTIONS\s*:\s*(\S.*)$')
COMMENT_RE = re.compile('(?:^|\s+)#.*$')
DEFAULT_RE = re.compile('(?!)')

# module methods
def load_regexps(input_file, encoding = 'utf8', close_file = True):
    '''Load regular expressions from text file.

    Read input file passed as argument and convert lines contained
    there to a RegExp union.'''
    cnt = 0
    # re_list will hold multiple lists each consisting of 2
    # elements. The fist element will be a list of regular
    # expressions and the second one will be a list of flags.
    # E.g.
    # re_list = [ [['papa', 'mama'], RE.UNICODE], [['sister'], RE.IGNORECASE]]
    re_struct = [[], 0]
    re_list = [re_struct]
    match = None
    for line in input_file:
        try:
            line = line.decode(encoding)
        except UnicodeDecodeError as e:
            e.reason += '\nIncorrect encoding (' + encoding + \
                ') specified for file ' + input_file.name
            raise e
        match = OPTIONS_RE.match(line)
        # different regexp options will separate different
        # chunks of regular expressions
        if match:
            # increment counter only if we have already seen any
            # regexps before
            if cnt != 0 or re_list[0][0]:
                cnt += 1
                re_list.append(re_struct)
                # convert options passed as strings to valid python
                # code
                re_list[cnt][1] = eval(match.group(1))
        else:
            # strip off comments
            line = skip_comment(line).strip()
            # and remember the line if it is not empty
            if line:
                re_list[cnt][0].append(line)
    input_file.close()

    if re_list:
        # unite regexps into groups
        re_list = [re.compile('(?:' + '|'.join(rexps) + ')', ropts) \
                       for (rexps, ropts) in re_list]
    else:
        # return a never matching regexp by default
        re_list = [DEFAULT_RE]
    # return a special object used to handle multiple regular
    # expressions
    return MultiRegExp(re_list)


def skip_comment(istring):
    '''Return input string with comments stripped-off.'''
    return COMMENT_RE.sub('', istring, 1)

# module classes
class MultiRegExp(list):
    '''Container class used to hold multiple compiled regexps.'''

    # overwriting default class
    def finditer(self, istring):
        '''Expanding default finditer operator.'''
        output = []
        groups = ()
        for re in self:
            # collect all possible matches as tuples for all regexps
            for match in re.finditer(istring):
                groups = match.groups()
                # iterate over all possible groups of re
                for gid in range(0, len(groups)):
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
        return MultiRegExpMatch(output)

        # to be continued...


class MultiRegExpMatch(list):
    '''Container class used to hold match objects produced by MultiRegExp.'''
    def __init__(self, matches=[]):
        '''Create an instance of MultiRegExpMatch.'''
        super(MultiRegExpMatch, self).__init__(self.select_llongest(matches))

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
            if end < prev_start:
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
        mcontainer1 = copy.copy(self)
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

##################################################################
