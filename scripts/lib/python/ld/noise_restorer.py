#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Module providing methods and classes for restoring removed noise elements.

Constants:
DEFAULT_NR_FILE - default file which contains a list of noise elements
                  that should be restored

Classes:
NoiseRestorer() - class for restoring noise information

"""

##################################################################
# Libraries
from ld import skip_comments, RuleFormatError
from alt_fio import AltFileInput
from replacements import Replacement
from offsets import Offsets

import os
import re

##################################################################
# Constants
DEFAULT_NR_FILE = ("{SOCMEDIA_ROOT}/lingsrc/noise_restorer"
                   "/elements2restore.txt").format(**os.environ)
__RWORD_RE__ = re.compile(r"""\s*"((?:[^"]|\\")+)"\s*\Z""")
__RREX_RE__ = re.compile(r"""\s*/((?:[^/]|\\/)+)/\s*\Z""")


##################################################################
# Class
class NoiseRestorer:
    """Class for restoring strings that were previosuly deleted from messages.

    Instance variables:

    self.rwords - set of replacement words that should be restored
    self.rre - list of regular expressions that match replacement elements
               which should be restored
    self.restoreList - list of elements that should be restored
    self.tokenOffsetList - list of token offsets
    self.t_offset - offset of the next token
    self.t_length - length of the next token
    self.r_offset - offset of the next element to be restored
    self.r_length - length of the next element to be restored

    Public methods:
    __init__() - initialize an instance of the class

    """

    def __init__(self, ifile=DEFAULT_NR_FILE):
        """Create an instance of NoiseRestorer.

        @param ifile - name of file containing list of elements which should be
                       restored

        """
        # set of words which are replecements that should be restored
        self.rwords = set([])
        # list of regexps, which are checked against replacements and once they
        # match these replecements, those replacements should be restored to
        # original form
        self.rre = []
        # container for storing replacement information
        self.restoreList = []
        self.tokenOffsets = Offsets()
        finput = AltFileInput(ifile)
        mobj = None
        for line in finput:
            line = skip_comments(line)
            if not line:
                continue
            mobj = __RWORD_RE__.match(line)
            if mobj:
                self.rwords.add(mobj.group(1))
                continue
            mobj = __RREX_RE__.match(line)
            if mobj:
                self.rre.append("(?:" + mobj.group(1) + ")")
                continue
            raise RuleFormatError(
                "Unrecognized line format for NoiseRestorer."
            )
        if self.rre:
            self.rre = re.compile("(?:" + '|'.join(self.rre) + ")")
        else:
            self.rre = re.compile("(?!)")
        self.t_offset = -1
        self.r_offset = -1
        self.t_length = -1
        self.r_length = -1

    def read_meta_line(self, istring):
        """
        Read and store necessary meta-information from istring.

        @param istring - input string to be parsed

        @return void

        """
        # if meta line represents an information
        mobj = Replacement.match(istring)
        if mobj:
            # create new replacement object on the basis of this match
            repl = Replacement(*mobj.groups())
            # check if this replacement corresponds to what we want to restore
            if repl.replacement in self.rwords or self.rre.match(
                    repl.replacement):
                self.restoreList.append(repl)
                self.restoreList.sort(key=lambda x: x.offset)
                self.r_offset, self.r_length = \
                    self.restoreList[0].offset, self.restoreList[0].length
            # print >> sys.stderr, repr(self.restoreList)
        else:
            # otherwise, try to parse line as if it had information about token
            # offsets
            self.tokenOffsets.parse(istring)

    def clear(self):
        """
        Clear information about word offsets and replacements.
        """
        del self.restoreList[:]
        self.tokenOffsets.clear()
        self.t_offset = -1
        self.r_offset = -1
        self.t_length = -1
        self.r_length = -1

    def restore(self, iline):
        """
        Restore line if necessary.

        @param iline - line to be restored

        @return line with restored noise elements

        """
        # if no words have to be restored, simply return the line unchanged
        # print >> sys.stderr, "iline = ", iline
        # print >> sys.stderr, "self.r_offset = ", self.r_offset
        if self.r_offset < 0:
            return iline
        # otherwise, get offset and length of the new token
        self.t_offset, self.t_length = self.tokenOffsets.popleft_token()
        # print >> sys.stderr, "self.t_offset = ", self.t_offset
        # print >> sys.stderr, "self.t_length = ", self.t_length
        # check that new token could be retrieved
        assert(self.t_offset != None)
        # check whether offset of the replacement lies within the boundaries of
        # the next token
        assert(self.t_offset <= self.r_offset)
        off_diff = self.r_offset - self.t_offset
        if off_diff < self.t_length:
            # append to the return line everything up to the beginning of the
            # replacement
            retline = iline[:off_diff]
            # add original string
            retline += self.restoreList[0].orig
            # if the length of the replacement is less than the length of the
            # token, then add the rest of the token to the `retline'
            rest_offset = off_diff + self.r_length
            if len(iline) > rest_offset:
                retline += iline[rest_offset:]
            # delete the inserted replacement
            self.__popleft_replacement__()
            return retline
        else:
            return iline

    def __popleft_replacement__(self):
        """
        Pop the first replacement and update self.r_offset and self.t_length.
        """
        lretlist = len(self.restoreList)
        if lretlist:
            self.restoreList.pop(0)

        if lretlist > 1:
            self.r_offset, self.r_length = self.restoreList[0].offset, self.restoreList[0].length
        else:
            self.r_offset = self.r_length = -1

        # et = parse_xml_line(istring)
        # if et:
        #     if et[0] == "word":
        #         if et[1]:
        #             self.offset = int(et[1].get("offset", -1))
        #             self.lngth  = int(et[1].get("length", -1))
        #         return ""
        #     elif et[0] == "replaced":
        #         if et[1]:
        #             repl = et[1].get("replace", None)
        #             if repl and (repl in self.rwords or self.rre.match(repl)):
        #                 ofs = int(et[1].get("offset", -1))
        #                 if not self.offsetList or ofs != self.offsetList[-1]:
        #                     self.offsetList.append(ofs)
        #                 if self.restoreInfoSet[ofs]:
        #                     self.restoreInfoSet[ofs][0] += int(et[1].get("length"))
        #                     self.restoreInfoSet[ofs][1] = self.restoreInfoSet[ofs][1] + \
        #                         et[1].get("orig")
        #                 else:
        #                     self.restoreInfoSet[ofs] = [int(et[1].get("length")), \
        #                                                     et[1].get("orig")]
        #         return ""
        #     elif et[0] == "sentence" :
        #         self.offsetList = []
        #         self.restoreInfoSet.clear()
        #         lngth  = None
        # elif self.offsetList and self.offset >= 0:
        #     if self.offsetList[0] >= self.offset and \
        #             self.offsetList[0] <= (self.offset + self.lngth):
        #         ostring = istring[:self.offsetList[0] - self.offset]
        #         ostring += self.restoreInfoSet[self.offsetList[0]][1]
        #         ostring += istring[(self.offsetList[0] - self.offset + \
        #                                 self.restoreInfoSet[self.offsetList[0]][0]):]
        #         self.offsetList = self.offsetList[1:]
        #         istring = ostring
        # self.offset = -1
        # self.lngth = -1
        # return istring
