#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
import re
import sys

from alt_fio import AltFileInput
from ..lingre.lre import RegExp
from ..lingre import RE_OPTIONS
from ..stringtools import upcase_capitalize
from .. import skip_comments, DEFAULT_RE, RuleFormatError
from .  import MAP_DELIMITER

##################################################################
# Class
class Map:
    """Class for replacing input text according to loaded rules."""

    def __init__(self, ifile = None, encd = 'utf8'):
        """Read map entries from ifile and transform them to a dictionary.

        Map entries will be read from input file ifile, which should have the
        form:

        src_entry \t trg_entry

        These entries will be transformed to a dict of form:

        map[src_entry] = trg_entry

        Additionally, a special regular expression will be generated combining
        all dict keys.
        """

        self.encd  = encd
        # both instance variables below will be populated in __load()
        self.map   = {}
        self.flags = ''
        self.ignorecase = False
        src = trg  = ''
        # load entries from ifile if it is specified
        if ifile:
            self.map = self.__load(ifile)
        # compile an RE capturing all source entries
        self.re = self.__compile_re(self.flags, self.map.keys())
        # define a function for adjustment of replacements
        if self.ignorecase:
            self.adjust_repl = lambda x: upcase_capitalize( \
                self.map[re.escape(x.lower())], x)
        else:
            self.adjust_repl = lambda x: self.map[re.escape(x)]

    def reverse(self, lowercase_key = False):
        """Return reverse copy of map."""
        # create an empty object of same class
        ret = self.__class__(None)
        # copy over all entries from self.map and swap key and value
        for src, trg in self.map.items():
            if lowercase_key:
                src = src.lower()
            if (trg in ret.map) and (src != ret.map[trg]):
                raise RuntimeError('''
Could not reverse map. Duplicate translation variants for '{:s}':
{:s} vs. {:s}'''.format(trg.encode('utf-8'), \
                            ret.map[trg].encode('utf-8'), \
                            src.encode('utf-8')))
            ret.map[trg] = src
        ret.re = self.__compile_re(ret.map)
        return ret

    def replace(self, istring, memory = None):
        """Replace all occurrences of src entries with trg in ISTRING.

        Search in ISTRING for all occurrences of src map entries and
        replace them with their corresponding trg form from self.map"""
        ostring = ''            # output string
        repl    = ''            # replacement
        start   = 0             # starting position
        mstart = mend = 0       # starting and ending position of replacement
        for mobj in self.re.finditer(istring):
            mstart, mend = mobj.span()
            ostring += istring[start:mstart]
            repl = self.adjust_repl(mobj.group(0))
            if memory:
                memory.update(mstart, len(repl) - (mend - mstart))
            ostring += repl
            start = mend
        ostring += istring[start:]
        return ostring

    # sub is an alias for replace
    sub = replace

    ##################
    # Private Methods
    def __load(self, ifile):
        """Load map entries from file ifile."""
        # load map entries from file
        output = {}
        optmatch = None
        finput = AltFileInput(ifile, encoding = self.encd)
        for line in finput:
            if line:
                optmatch = RE_OPTIONS.match(line)
                if optmatch:
                    if self.flags:
                        raise  RuleFormatError( \
                            msg = "Multiple flag lines are not supported", \
                                efile = finput)
                    else:
                        self.flags = optmatch.group(1)
                        self.ignorecase = RegExp(self.flags, "").re.flags & re.IGNORECASE
                        continue
                # find map entries
                line = skip_comments(line)
                m = MAP_DELIMITER.search(line)
                if m:
                    src, trg = self.__normalize_quotes(line[0:m.start()], \
                                                           line[m.end():])
                    if not (src and trg):
                        print src.encode('utf-8')
                        print trg.encode('utf-8')
                        raise RuleFormatError(efile = finput)
                    src = re.escape(src)
                    if self.ignorecase:
                        output[src.lower()] = trg
                    else:
                        output[src] = trg
                elif line:
                    raise RuleFormatError(efile = finput)
        return output

    def __normalize_quotes(self, *istrings):
        """Normalize quotes at begin and end of istring.

        Unescaped qutes at the beginning and and of a string are
        stripped and escaped quotes with literal ones E.g.

        "Hello World" --> Hello World
        \"Hello World\" --> "Hello World"

        NOTE: only a single occurrence of unescaped quotes is
        stripped.
        """
        output = []
        for istring in istrings:
            lngth = len(istring)
            quotes = istring[0] + istring[-1]
            esc_quotes = istring[0:2] + istring[-2:]
            if lngth > 1 and (quotes == "''" or quotes == '""'):
                istring = istring[1:-1]
            elif lngth > 3 and (esc_quotes == r"\'\'" or esc_quotes == r'\"\"'):
                istring = istring[1:-2] + istring[-1:]
            output.append(istring)
        return output

    def __compile_re(self, flags = '', rules = []):
        """Compile RE according to flags combining all rules using `|'."""
        if not rules:
            return DEFAULT_RE
        regexp = RegExp(flags, *rules).re
        return regexp
