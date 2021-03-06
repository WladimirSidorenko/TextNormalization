#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Module providing methods and classes for manipulating regular and misspelled
umlaut characters.
"""

##################################################################
# Libraries
import ld.stringtools
from ld.lingre  import lre
from ld.lingmap import lmap
from replacements import Memory

import re
import warnings

##################################################################
# Constants

##################################################################
# Classes
class Umlaut:
    """Class containing all relevant information regarding umlauts."""

    UMLAUTS_UPPER = ["Ä", "Ö", "Ü"]
    UMLAUTS_LOWER = ["ä", "ö", "ü"]
    UMLAUTS = UMLAUTS_UPPER + UMLAUTS_LOWER
    UMLAUTS_RE = re.compile(('(?:' + '|'.join(UMLAUTS) + ')'), re.UNICODE)

    def __initialize__(self):
        """This class only has constants so far."""
        pass

class UmlautRestorer(Umlaut):
    """Class for restoration of misspelled umlauts."""

    # all 3 arguments are supposed to be open file descriptors, which
    # will be closed at the end, if function argument close_fd is True
    # (the default).
    def __init__(self, misspelled_re_f, missp2correct_f, \
                           exceptions_f = None, close_fd = True):
        """Create an instance on Umlaut class. """
        self.misspelled_re     = lre.MultiRegExp(misspelled_re_f)
        self.missp2correct_map = lmap.Map(missp2correct_f)
        self.correct2missp_map = self.missp2correct_map.reverse(lowercase_key = True)
        self.exceptions    = UmlautExceptions(exceptions_f, close_fd = close_fd, \
                                                  reverse_map = self.correct2missp_map)

    def missp2correct(self, istring, memory):
        """Restore ae, oe, ue in input string to umlaut characters.

        Search for all occurrences of exceptional substrings where
        writings like "ae" etc. are legal and replace all in-between
        occurrences of "ae" and friends with corresponding correct
        umlaut characters from self.missp2correct_map."""
        ostring = ''
        str_start = 0    # start character of istring
        # find all exceptional substring (lowercased version of string is used)
        except_spans = self.exceptions.re.finditer(istring.lower())
        # iterate over all exceptional spans
        for (except_start, except_end) in except_spans:
            # print >> sys.stderr, "except_start:", str(except_start)
            # print >> sys.stderr, "except_end:", str(except_end)
            # print >> sys.stderr, "str_start:", str(str_start)
            # if debug:
            #     print >> sys.stderr, "Exception: ", istring[except_start:except_end].encode("utf-8")
            # restore umlauts before match
            ostring += self.__restore_(istring[str_start:except_start], str_start, memory)
            # restore umlauts in matched exception (yes, sometimes we need it
            # too), e.g. in `aufzuerw<ae>rmen' --> `aufzuerw<ä>rmen'
            ostring += self.__restore_exception_(istring[except_start:except_end])
            # and assume that the istring starts with the end of the current
            # match
            str_start = except_end
        # restore umlauts in the rest of the input string
        ostring += self.__restore_(istring[str_start:], str_start, memory)
        return ostring

    def correct2missp(self, istring):
        """Replace all occurrences of umlauts Ä, Ö, Ü with AE, OE, UE."""
        raise NotImplementedError

    def __restore_(self, istring, str_start, memory):
        """Unconditionally replace misspelled umlauts."""
        # initialize variables
        ostring = repl = ''
        lenrepl = lenorig = start = 0
        # print >> sys.stderr, "*str_start:", str_start
        # find all occurrences of misspelled umlauts in ISTRING
        for (missp_start, missp_end) in self.misspelled_re.finditer(istring):
            # add to the resulating string everything to the left of the 1-st
            # occurrence of incorrectly spelled umlaut
            ostring += istring[start:missp_start]
            # devise a replacement
            repl = self.__restore_helper_(istring[missp_start:missp_end])
            # calculate how different the length of the new replacement is from
            # the length of the replaced subtext
            lenrepl, lenorig = len(repl), missp_end - missp_start
            if lenrepl != lenorig:
                memory.update(pos = str_start + missp_start, \
                                  delta = lenrepl - lenorig)
            # add the replacement to the resulting output string
            ostring += repl
            # remember the new starting position
            start = missp_end
        # add everything to the right of the last occurred replacement to the
        # resulting substring
        ostring += istring[start:]
        # return the result
        return ostring

    def __restore_helper_(self, key):
        """Yield proper replacement for given misspelled sequence."""
        # look-up the key in replacement dictionary
        if key in self.missp2correct_map.map:
            return self.missp2correct_map.map[key]
        # the situation when the key is not present in replacement map
        # should never occur, but if it does - issue a warning and return
        # matched substring unmodified
        else:
            warnings.warn("No replacement found for '{:s}'.".format(key), \
                              warnings.RuntimeWarning)
            return key


    def __restore_exception_(self, istring):
        """Replace misspelled umlauts."""
        return self.exceptions.restore(istring)


class UmlautExceptions(Umlaut):
    """Class of words in which writings ae, oe, ue are legal."""

    def __init__(self, exception_file = None, \
                     close_fd = True, reverse_map = lmap.Map()):
        """Create an instance of UmlautExceptions."""
        self.missp2correct = {}
        self.re = lre.MultiRegExp(exception_file, no_inner_groups = True, \
                                      istring_hook = (lambda istring: \
                                                          self.__read_exception_(istring, reverse_map)))

    def restore(self, istring):
        """Restore any still valid umlauts in a spoiled exception."""
        key = istring.lower()
        if key in self.missp2correct:
            return ld.stringtools.adjust_case(self.missp2correct[key], istring)
        return istring

    def __read_exception_(self, istring, reverse_map):
        """Spoil umlauts in ISTRING and return original and spoiled str."""
        # generating all possible spoiled variants, i.e. 2^(# of umlauts) would
        # be more robust, but in practice it will rather slow things down
        result = istring.lower()
        spoiled = reverse_map.sub(result) # replace any exisiting umlauts with
                                          # their spoiled variant
        if result != spoiled:
            self.missp2correct[spoiled] = istring
            return [result, spoiled]
        else:
            return [result]
