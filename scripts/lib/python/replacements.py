#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Description
"""Module for handling information about token offsets.

   Constants:

   Classes:
   Offsets  - class for storing and outputting offset information

"""

##################################################################
# Modules
import bisect
import os
import re
import sys

##################################################################
# Constants
# __REPL_TAG_XXX__ - is just a common template for both REPL_TAG_STR and
# REPL_TAG_RE
__REPL_TAG_XXX__ = os.environ.get("SOCMEDIA_ESC_CHAR", "") + \
    "\treplace\toffset\t{:s}\tlength\t{:s}\tnum\t{:s}\torig\t{:s}\treplacement\t{:s}\tinfo_end"
REPL_TAG_STR = unicode(__REPL_TAG_XXX__.format(r"{:d}", r"{:d}", r"{:d}", r"{:}", r"{:s}"))
REPL_TAG_RE  = re.compile(__REPL_TAG_XXX__.format(r"(\d+)", r"(\d+)", r"(\d+)", '([^\t]*)', \
                                                      '([^\t]*)'))

##################################################################
# Classes
class Replacement:
    """
    This class provides methods and structures for storing and updating
    information about offsets of one particular replacement.

    Instance Variables:
    self.offset - position in string at which replacement took place
    self.length - length of inserted replacement
    self.orig   - original word that was replaced
    self.replacement - word that was inserted as replacement

    Class Method:
    match()     - parse given input string according to `re' specification
                  for repl

    Instance Methods:
    __init__() - initialize instance variables
    __str__()  - return string representation of this object

    """

    @classmethod
    def match(cls, istring):
        """Parse string according to REPL_TAG_RE and return match object on success."""
        return REPL_TAG_RE.match(istring)

    def __init__(self, offset = 0, length = 0, num = 0, orig = '', replacement = ''):
        """Initialize instance variables."""
        self.offset = int(offset)
        self.length = int(length)
        self.num    = int(num)
        self.orig   = orig
        self.replacement = replacement

    def __str__(self):
        """Return printable string representation of this object."""
        return REPL_TAG_STR.format(self.offset, self.length, self.num, \
                                       self.orig, self.replacement)

##################################################################
class Memory:
    """
    This class provides methods and structures for storing and updating
    information about all replacements.

    Instance Variables:
    self.replnum      - counter of made replacements
    self.replacements - array of remembered replacements

    Methods:
    __init__() - initialize instance variables
    append()   - construct a replacement and append it to `self.replacements'
    forget_all() - forget all memory information
    is_empty() - return boolean if any records are stored
    update()   - shift positions of elements in `self.replacements'
    __str__()  - return string representation of this object
    """

    def __init__(self):
        """Initialize instance variables."""
        self.replnum = 0
        self.replacements = []
        self.__offset_list__ = []
        self.__offset2pos__  = {}

    def append(self, *args, **kwargs):
        """Construct Replacement from arguments and store it as tuple."""
        newrepl = Replacement(*args, num = self.replnum, **kwargs)
        self.replnum += 1
        self.__store_repl__(newrepl)

    def parse(self, iline):
        """Parse input line and store information about described replacement."""
        # first determine whether given input line describes a replacement
        mobj = Replacement.match(iline)
        if mobj:
            # if it does, store the information about the replacement in self
            self.__store_repl__(Replacement(*mobj.groups()))
        return mobj

    def is_empty(self):
        """Return true if memory contains any memory records."""
        # first determine whether given input line describes a replacement
        return (self.replnum == 0)

    def update(self, pos, delta):
        """Update offsets of all elements in `self.replacements' if needed."""
        if not delta:
            return
        new_offset = 0
        # find the rightmost value in `self.__offset_list__' which is greater
        # than `pos'
        idx = self.__find_gt__(self.__offset_list__, pos)
        # if no element in `self.__offset2pos__' is greater than `pos', just
        # return - we do not have to do anything
        if idx == len(self.__offset_list__):
            return
        # Otherwise, for all elements following `pos' change them by `delta'.
        #
        # Note, that if delta is greater than zero, we should iterate from the
        # end, if its less than zero, we should iterate from the beginning.
        offset_list = self.__offset_list__[idx:]
        # print >> sys.stderr, "Updating at pos =", pos, " with delta =", delta
        # print >> sys.stderr, "self.__offset_list__[idx:]", repr(self.__offset_list__[idx:])
        # print >> sys.stderr, "self.__offset2pos__", repr(self.__offset2pos__)
        if delta > 0:
            offset_idx = 0
            offset_list.reverse()
            for offset in offset_list:
                offset_idx -= 1
                self.__update__(offset, delta, offset_idx)
        else:
            offset_idx = idx
            for offset in offset_list:
                self.__update__(offset, delta, offset_idx)
                offset_idx += 1
        # print >> sys.stderr, "After update: self.__offset_list__[idx:]", repr(self.__offset_list__[idx:])
        # print >> sys.stderr, "After update: self.__offset2pos__", repr(self.__offset2pos__)

    def forget_all(self):
        """Erase all information which is currently in memory."""
        self.replnum = 0
        del self.replacements[:]
        del self.__offset_list__[:]
        self.__offset2pos__.clear()

    def __iter__(self):
        """Standard method for iterator protocol."""
        return self.replacements.__iter__()

    def __str__(self):
        """Return printable string representation of this object."""
        # print >> sys.stderr, "Offset2Pos:", repr(self.__offset2pos__)
        # print >> sys.stderr, "Offset list:", repr(self.__offset_list__)
        return u'\n'.join([unicode(r) for r in self.replacements])

    def __store_repl__(self, repl):
        """Store replacement in underlying container."""
        # determine, to which place we are going to insert the next replacement
        idx = len(self.replacements)
        # check if we already have a replacement for this position
        if repl.offset in self.__offset2pos__:
            # if we do, remember that there is one more replacement for this
            # position
            self.__offset2pos__[repl.offset].append(idx)
        else:
            # otherwise, remember this new replacement
            self.__offset2pos__[repl.offset] = [idx]
            bisect.insort(self.__offset_list__, repl.offset)
        # store the replacement
        self.replacements.append(repl)
        self.replnum += 1

    def __find_gt__(self, arr, pos):
        """Find index of the first element in __offset_list__ greater than
        pos."""
        return bisect.bisect_right(arr, pos)

    def __update__(self, old_offset, delta, offset_idx):
        """Update offsets in offset list and offset2pos dictionary."""
        # calculate new offset by adding delta
        # print >> sys.stderr, "old_offset = ", old_offset, "delta = ", delta, \
        # "offset_idx = ", offset_idx
        new_offset = old_offset + delta
        # print >> sys.stderr, "self.__offset_list__ = ", repr(self.__offset_list__)

        # iterate over each replacement corresponding to given offset and
        # update their information
        for repl_idx in self.__offset2pos__[old_offset]:
            # update `self.replacements'
            self.replacements[repl_idx].offset = new_offset
        # update `__offset_list__'
        del self.__offset_list__[offset_idx]
        # update `__offset2pos__'
        if new_offset in self.__offset2pos__:
            self.__offset2pos__[new_offset] += self.__offset2pos__[old_offset]
        else:
            self.__offset2pos__[new_offset] = self.__offset2pos__[old_offset]
            # insert new offset in offset_list
            bisect.insort(self.__offset_list__, new_offset)
        # remove obsolete offset from `self.__offset2pos__'
        self.__offset2pos__.pop(old_offset, None)

##################################################################
class Restorer:
    """
    This class provides methods for restoring elements based on their offsets.

    Constants:

    Instance Variables:

    Methods:
    __init__() - initialize instance variables
    replace()  - construct a replacement and append it to `self.replacements'
    update()   - shift positions of elements in `self.replacements'
    __str__()  - return string representation of this object
    """

    DEFAULT_RULE_FILE = "{SOCMEDIA_ROOT}/lingsrc/noise_restorer/elements2restore.txt".format(**os.environ)
    RWORD_RE = re.compile(r"""\s*"((?:[^"]|\\")+)"\s*\Z""")
    RREX_RE  = re.compile(r"""\s*/((?:[^/]|\\/)+)/\s*\Z""")

    def __init__(self, ifile = DEFAULT_RULE_FILE):
        """Initialize NoiseRestorer, read entries to restore from
        DEFAULT_RULE_FILE"""
        # set of words which are replecements that should be restored
        self.rwords = set([])
        # list of regexps, which are checked against replacements and once they
        # match these replecements, those replacements should be restored to
        # original form
        self.rre    = []
        # container to store replacement information
        self.offsetList      = list()
        self.restoreInfoSet  = defaultdict(list)
        finput = AltFileInput(ifile)
        mobj = None
        for line in finput:
            line = skip_comments(line)
            if not line:
                continue
            mobj = RWORD_RE.match(line)
            if mobj:
                self.rwords.add(mobj.group(1))
                continue
            mobj = RREX_RE.match(line)
            if mobj:
                self.rre.append("(?:" + mobj.group(1) + ")")
                continue
            raise RuleFormatError("Unrecognized line format for NoiseRestorer.")
        if self.rre:
            self.rre = re.compile("(?:" + '|'.join(self.rre) + ")")
        else:
            self.rre = re.compile("(?!)")
        self.offset = -1
        self.length = -1

    def replace(self, istring):
        """Read istring and return it unmodified or its replaced copy."""
        et = parse_xml_line(istring)
        if et:
            if et[0] == "word":
                if et[1]:
                    self.offset = int(et[1].get("offset", -1))
                    self.lngth  = int(et[1].get("length", -1))
                return ""
            elif et[0] == "replaced":
                if et[1]:
                    repl = et[1].get("replace", None)
                    if repl and (repl in self.rwords or self.rre.match(repl)):
                        ofs = int(et[1].get("offset", -1))
                        if not self.offsetList or ofs != self.offsetList[-1]:
                            self.offsetList.append(ofs)
                        if self.restoreInfoSet[ofs]:
                            self.restoreInfoSet[ofs][0] += int(et[1].get("length"))
                            self.restoreInfoSet[ofs][1] = self.restoreInfoSet[ofs][1] + \
                                et[1].get("orig")
                        else:
                            self.restoreInfoSet[ofs] = [int(et[1].get("length")), \
                                                            et[1].get("orig")]
                # print repr(self.restoreInfoSet)
                return ""
            elif et[0] == "sentence" :
                # print >> sys.stderr, ' '.join([str(i) for i in self.offsetList])
                # print >> sys.stderr, repr(self.restoreInfoSet)
                self.offsetList = []
                self.restoreInfoSet.clear()
                lngth  = None
        elif self.offsetList and self.offset >= 0:
            if self.offsetList[0] >= self.offset and \
                    self.offsetList[0] <= (self.offset + self.lngth):
                # print >> sys.stderr, "triggered"
                ostring = istring[:self.offsetList[0] - self.offset]
                # print >> sys.stderr, "Istring: ", istring
                # print >> sys.stderr, repr(self.offsetList)
                # print >> sys.stderr, repr(self.offset)
                ostring += self.restoreInfoSet[self.offsetList[0]][1]
                ostring += istring[(self.offsetList[0] - self.offset + \
                                        self.restoreInfoSet[self.offsetList[0]][0]):]
                self.offsetList = self.offsetList[1:]
                istring = ostring
        self.offset = -1
        self.lngth = -1
        return istring
