#!/usr/bin/env python2.7

"""Initialize original argparse parser and set up common options.

This module wraps up original argparse library by initializing its parser and
setting up some options and arguments which are common to all the scripts which
include the present module. Additionally, two more methods
`.add_file_argument()' and `.add_rule_file_argument()' are added to
argparse.ArgumentParser()

Members of this module:
alt_argparse.argparser  - reference to initialized instance of
                          argparse.ArgumentParser()
alt_argparse.*          - access to all the other original argparse members

Initialized options:
-f, --flush             flush output

-e, --encoding          input/output encoding

-s, --skip-line         line to be skipped during processing

files                   input files to be processed

Class:
AltArgumentParser() - successor of argparse.ArgumentParser() extending its
                      parent with some methods

  self.add_file_argument() - wrapper around parental .add_argument() method
                          explicitly trimmed for adding file arguments.

  self.add_rule_file_argument() - wrapper around parental .add_argument()
                          method explicitly trimmed for adding file arguments
                          with linguistic rules. (depending on whether version
                          environment is set or not, this argument will be
                          optional or mandatory)

"""

##################################################################
# Import modules
from argparse import *
import os

##################################################################
# Declare interface
__all__ = ["DEFAULT_LANG", "AltArgumentParser", "argparser"]

##################################################################
# Variables
# I/O encoding can be specified via this variable. Its value is determined by
# inspecting following variables in that order: SOCMEDIA_LANG, LC_ALL,
# LC_CTYPE, LANG. If none of these variables is set - utf-8 will be chosen by
# default.
DEFAULT_LANG = "utf-8"
for langvar in ["SOCMEDIA_LANG", "LC_ALL", "LC_CTYPE", "LANG"]:
    if langvar in os.environ:
        lang = os.environ[langvar]
        last_dot = str.rfind(lang, '.')
        if last_dot > -1:
            lang = lang[last_dot + 1:]
        DEFAULT_LANG = lang
        break

##################################################################
# Subclass ArgumentParser() and extend its successor with new methods.
class AltArgumentParser(ArgumentParser):
    """Class extending standard ArgumentParser() with 2 methods."""

    def add_file_argument(self, *args, **kwargs):
        """Wrapper around add_argument() method explicitly dedicated to files.

        This method simply passes forward its arguments to add_argument()
        method of ArgumentParser() additionally specifying that the type of
        argument being added is a readable file."""
        return self.add_argument(*args, type = FileType(mode = 'r'), **kwargs)

    def add_rule_file_argument(self, *args, **kwargs):
        """explicitly dedicated to rule files.

        This method checks if file pointed by default argument exist and if it
        does this file will be regarded as default argument. If default file
        doe not exist, the option and its argument will be considered
        mandatory."""
        if "required" in kwargs or "default" in kwargs:
            return self.add_argument(*args, **kwargs)
        d = kwargs.pop("dir", "")
        if d and d[-1] != '/':
            d = d + '/'
        f = (d + kwargs.pop("file", "")).format(**os.environ)
        if f and os.path.isfile(f):
            return self.add_argument(*args, type = FileType(mode = 'r'), default = f, **kwargs)
        else:
            return self.add_argument(*args, type = FileType(mode = 'r'), required = True, **kwargs)

##################################################################
# Set up an argument parser
argparser = AltArgumentParser()
argparser.add_argument("-f", "--flush", help="flush output", action="store_true")
argparser.add_argument("-e", "--encoding", help="input/output encoding", \
                           default = DEFAULT_LANG)
argparser.add_argument("-s", "--skip-line", help="""line to be skipped during
processing (only encoding/decoding and strip() operations will be applied to
that line)""")
argparser.add_argument("files", help="input files", nargs = '*', metavar="file")
