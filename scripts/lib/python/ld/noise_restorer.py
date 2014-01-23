#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Module providing methods and classes for restoring removed noise strings.

Constants:
DEFAULT_NR_FILE    - default file which contains a list of noise elements
                     that should be restored

Classes:
NoiseRestorer()    - class for restoring noise information

"""

##################################################################
# Libraries
from ld import skip_comments, RuleFormatError
from ld.stringtools import parse_xml_line

import os
from collections import defaultdict

##################################################################
# Constants
DEFAULT_NR_FILE = "{SOCMEDIA_ROOT}/lingsrc/noise_restorer/elements2restore.txt".format(**os.environ)
__RWORD_RE__ = re.compile(r"""\s*"((?:[^"]|\\")+)"\s*\Z""")
__RREX_RE__  = re.compile(r"""\s*/((?:[^/]|\\/)+)/\s*\Z""")

##################################################################
# Class
class NoiseRestorer:
    """Class for restoring strings that were previosuly deleted from message.

    Instance variables:

    Public methods:
    __init__() - initialize an instance of the class

    """

    def __init__(self, ifile = DEFAULT_RULE_FILE):
        """Create an instance of NoiseRestorer.

        @param ifile - name of file containing list of elements which should be
                       restored

        """
        # set of words which are replecements that should be restored
        self.rwords = set([])
        # list of regexps, which are checked against replacements and once they
        # match these replecements, those replacements should be restored to
        # original form
        self.rre    = []
        # container for storing replacement information
        self.offsetList      = list()
        self.restoreInfoSet  = defaultdict(list)
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
                return ""
            elif et[0] == "sentence" :
                self.offsetList = []
                self.restoreInfoSet.clear()
                lngth  = None
        elif self.offsetList and self.offset >= 0:
            if self.offsetList[0] >= self.offset and \
                    self.offsetList[0] <= (self.offset + self.lngth):
                ostring = istring[:self.offsetList[0] - self.offset]
                ostring += self.restoreInfoSet[self.offsetList[0]][1]
                ostring += istring[(self.offsetList[0] - self.offset + \
                                        self.restoreInfoSet[self.offsetList[0]][0]):]
                self.offsetList = self.offsetList[1:]
                istring = ostring
        self.offset = -1
        self.lngth = -1
        return istring
