#!/usr/bin/env python2.7

"""Module for interactive communication with Hunspell.

Current official python egg `pyhunspell' hasn't been actively maintained for a
relatively long time and can't be properly installed via pip any more. Existing
replacement packages such as https://github.com/smathot/pyhunspell are only
adapted to some particular Unix systems (only Debian to be exact) and heavily
depend on their packages. This module is intended to provide an alternative
interface for communication with hunspell via an interactive pipe. It is
assumed to be platform-independent and requires only an existing hunspell
installation on the machine.

Constants:

DEFAULT_ENCD - default encoding used to communicate with hunspell (currently
                      set to UTF-8, but may be overridden during initialization
                      of Hunspell())
ESCAPE_CHAR - prefix character, which is prepended to each word being checked
                      in order to prevent possible special interpretation of
                      characters like %, &, etc. See hunspell's manual for
                      further reference.

VALID_WORD_MARKERS - string of characters which indicate a valid word in
                      hunspell's output. Its current value is "*+-"
SUGGESTIONS_START - a character which marks beginning of suggestions
SUGGESTIONS_DELIM - a regexp marking boundaries between suggestions
OUTPUT_DELIM - compiled regexp representing newline character which serves as
                      delimiter of multiple output lines

Classes:
Hunspell() - main class of this module. Establishes an interactive pipe
                      connected to hunspell.

Requires:
IPopen - custom library for interactive communication with pipes

"""

##################################################################
# Loaded Modules
import re
from ipopen import IPopen

##################################################################
# Constants
DEFAULT_ENCD = "UTF-8"
DEFAULT_LANG = "de_CH"
ESCAPE_CHAR = "^"
VALID_WORD_MARKERS = "*+-"
SUGGESTIONS_START = ':'
SUGGESTIONS_DELIM = re.compile(r",\s+")
OUTPUT_DELIM = re.compile(r'\n')
EXCEPTIONS = re.compile(r"\s*(?:hab)\s*\Z", re.I | re.L)
VALID_WORDS = set(["polit", "internas"]) # all words should be lowercased here


##################################################################
# Class
class Hunspell:

    """Class for interactive communication with hunspell program.

    An instance of this class establishes an interactive pipe to hunspell
    program. After that you can easily pass data to hunspell and read back and
    interpret its output from within your python script.

    This class provides following instance variables:
    self.encd      - encoding for communication with hunspell
    self.processor - an interactive subprocess pipe to hunspell
    self.version   - the version of hunspell program in use

    This class provides following public methods:
    __init__()     - initialize the afore-mentioned variables and establish
                     a pipe subprocess
    self.spell()    - check an input word on validness
    self.check()    - an alias for `spell()` method
    self.suggest()  - offer a list of possible suggestions for an unknown word
    self.close()    - close the streams associated with pipe subprocess

    """

    def __init__(self, encd=DEFAULT_ENCD, dic="de_CH", *hs_args):
        """Establish pipe to hunspell program.

        """
        self.encd = encd
        self.processor = IPopen(args=["hunspell", "-H", "-i", self.encd,
                                      "-d", dic] + list(hs_args),
                                skip_line='\n', skip_line_expect='\n',
                                thread=False, timeout=20)
        # After invocation, hunspell outputs a line with its version
        # number. Store this line in an instance variable for the case, that
        # anybody would like to see its value.
        self.version = self.processor.stdout.readline().strip()
        # private variable used for communication between spell() and suggest()
        self.__output__ = ''

    def spell(self, iword):
        """Check if iword is a valid word and return a bool."""
        iword = iword.strip()
        # The 1-st character returned from output will indicate whether the
        # word is valid or not. Note, that all encoding/decoding operations on
        # iword will be done in IPopen implicitly. Additionally, each word is
        # prefixed with a '^', since hunspell's manual states, that it is
        # recommended practice in order to prevent unexpected interpretation of
        # special characters.
        self.__output__ = self.processor.communicate(ESCAPE_CHAR + iword,
                                                     encd=self.encd)
        # If the 1-st returned character is among identifiers of valid strings,
        # return True, otherwise return False.
        return (((self.__output__ and self.__output__[0] in VALID_WORD_MARKERS)
                and not EXCEPTIONS.match(iword))
                or iword.lower() in VALID_WORDS)

    # method alias
    check = spell

    def spell_list(self, iwords):
        """Check if any of iwords is a valid and return a list of valid words.

        """
        l1 = len(iwords)
        # pass all the words to hunspell, and obtain its output as a single
        # string
        self.__output__ = self.processor.communicate(
            ESCAPE_CHAR + ' '.join(iwords), encd=self.encd).rstrip()
        # split this string on newlines and check whether resulting list has
        # the same length as list of input words
        result = OUTPUT_DELIM.split(self.__output__)
        # if we could not align both lists - return empty
        if len(result) != l1:
            return []
        else:
            return [iword for iword, res in zip(iwords, result) if
                    res and res[0] in VALID_WORD_MARKERS]

    def suggest(self, iword):
        """Return a list of spelling suggestions for misspelled word iword."""
        suggestions = []
        if self.spell(iword):
            return [iword]
        else:
            if SUGGESTIONS_START in self.__output__:
                start = self.__output__.index(SUGGESTIONS_START) + 1
                suggestions = SUGGESTIONS_DELIM.split(self.__output__[start:])
            return suggestions

    def close(self):
        """Close streams associated with underlying hunspell subprocess."""
        self.__output__ = ''
        self.processor.close()
