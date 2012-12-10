#!/usr/bin/env python2.7

'''
Initialize original argparse parser and set up common options.

This module wraps up original argparse library by initializing its parser and
setting up some options and arguments which are common to all the scripts which
include the present module. Additionally one more method (.add_file_argument())
is added to argparse.ArgumentParser()

Members of this module:
alt_argparse.argparser  - reference to initialized instance of
                          argparse.ArgumentParser()
alt_argparse.*          - access to all the other original argparse members

Initialized options:
-f, --flush             flush output
-s, --skip-line         line to be skipped during processing
files                   input files for processing

Additional classes:
AltArgumentParser()     - successor of argparse.ArgumentParser() extending
                          its parent with some methods
  self.add_file_argument()  - wrapper around parental .add_argument() method
                          explicitly trimmed for adding file arguments.
'''

##################################################################
# Import modules
from argparse import *
import sys as __sys__

##################################################################
# Declare interface
__all__ = ['AltArgumentParser', 'argparser']

##################################################################
# Subclass ArgumentParser() and extend its successor with new methods.
class AltArgumentParser(ArgumentParser):
    '''Class extending standard ArgumentParser() with some methods.'''

    def add_file_argument(self, *args, **kwargs):
        '''Wrapper around add_argument() method explicitly dedicated to files.

        This method simply passes forward its arguments to add_argument()
        method of ArgumentParser() additionally specifying that the type of
        argument being added is a readable file.'''
        return self.add_argument(*args, type = FileType(mode = 'r'), **kwargs)

##################################################################
# Set up an argument parser
argparser = AltArgumentParser()
argparser.add_argument('-f', '--flush', help='flush output', action='store_true')
argparser.add_argument('-s', '--skip-line', help='line to be skipped during processing')
argparser.add_argument('files', help='input files', nargs = '*', metavar='file')

# argparser.add_argument('files', help='input files', nargs = '*', \
#                            type = FileType(mode = 'r'), \
#                            default = [__sys__.stdin], metavar='file')
