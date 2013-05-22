#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""A light-weight alternative to standard fileinput library."""

##################################################################
# Loaded Modules
import sys
from os import getenv
from fileinput import *
from ld.stringtools import is_xml_tag
from alt_argparse import DEFAULT_LANG

##################################################################
# Interface
__all__ = ['AltFileInput', 'AltFileOutput']

##################################################################
# Constants
DEFAULT_INPUT  = sys.stdin
DEFAULT_OUTPUT = sys.stdout

##################################################################
# Class AltFileInput
class AltFileInput:
    """Class for reading and appropriate decoding of input strings."""
    def __init__(self, *ifiles, **kwargs):
        """Create an instance of AltFileInput."""
        # set up input encoding - use environment variable
        # SOCMEDIA_LANG or 'utf-8' by default
        self.encoding = kwargs.get('encoding', DEFAULT_LANG)
        # specify how to handle characters, which couldn't be decoded
        self.errors   = kwargs.get('errors', 'strict')
        # if skip_line was specified, establish an ad hoc function, which will
        # return true if its arg is equal
        if 'skip_line' in kwargs:
            self.skip = lambda line: line == kwargs['skip_line']
        else:
            self.skip = lambda line: False
        # if skip_xml was specified, establish an ad hoc function, which will
        # return true if its argument line is an XML tag
        if 'skip_xml' in kwargs:
            self.skip_xml = is_xml_tag # note, it's already a function
        else:
            self.skip_xml = lambda line: False
        # associate a print function with current fileinput, so that any input
        # lines, which should be skipped, could be sent to it
        if 'print_func' in kwargs:
            self.print_func = kwargs['print_func']
        else:
            # otherwise, standard print function will be used, however we
            # provide for a possibility, to specify the print destination via
            # 'print_dest' kwarg, so that even standard print function could be
            # easily re-directed
            if 'print_dest' in kwargs:
                self.print_dest = kwargs['print_dest']
            else:
                self.print_dest = DEFAULT_OUTPUT
            self.print_func = self.__print_func_
        #allow ifiles to appear both as list and as a
        # kw argument
        if not ifiles:
            ifiles = kwargs.get('ifiles', [DEFAULT_INPUT])
        # setting up instance variables
        self.files = ifiles     # list of input files
        self.fcnt  = -1         # counter for files
        self.current_file = None # file currently being read
        self.filename = None     # name of the file as a string
        self.fnr = 0             # current record number in the current file
        self.nr = 0              # number of records processed since
                                 # the beginning of the execution
        self.line = ''           # last line read-in
        # going to the 1-st file
        self.__next_file_()

    def next(self):
        """Yield next line or stop iteration if input exhausted."""
        self.line = self.current_file.readline()
        # print repr(self.line)
        if not self.line:
            self.__next_file_()
            return self.next()
        self.fnr +=1
        self.nr  +=1
        self.line = self.line.decode(encoding = self.encoding, \
                                         errors = self.errors).rstrip()
        # If the line read should be skipped, print this line and read the next
        # one.
        if self.skip(self.line) or self.skip_xml(self.line):
            self.print_func(self.line)
            return self.next()
        else:
            return self.line

    def __iter__(self):
        """Standard method for iterator protocol."""
        return self

    def __stop__(self):
        """Unconditionally raise StopIteration() error."""
        raise StopIteration

    def __next_file_(self):
        """Switch to new file if possible and update counters."""
        # close any existing opened files
        if self.current_file:
            self.current_file.close()
        # increment counter
        self.fcnt += 1
        # didn't calculate len() in __init__ for the case that
        # self.files changes somewhere in the middle
        if self.fcnt < len(self.files):
            # reset counters
            self.current_file = self.__open__(self.files[self.fcnt])
            self.filename = self.current_file.name
            self.fnr = 0
            self.line = ''
        else:
            # if we have exhausted the list of available files, all
            # subsequent calls to self.next will promptly redirect to
            # another functon which will unconditionally raise
            # a StopIterantion error
            self.next = self.__stop__
            self.next()

    def __open__(self, ifile):
        """Determine type of ifile argument and open it appropriately."""
        # Duck-Typing in real world - no matter what the object's name is, as
        # far as it provides the necessary method
        if hasattr(ifile, 'readline'):
            # file is already open
            return ifile
        elif isinstance(ifile, str) or \
                isinstance(ifile, buffer):
            if ifile == '-':
                return DEFAULT_INPUT
            # open it otherwise
            return open(ifile, 'r')
        else:
            raise TypeError('Wrong type of argument')

    def __print_func_(self, oline = ""):
        """Private function for outputting oline to particular destination stream."""
        print >> self.dest, oline


##################################################################
# Class AltFileOutput
class AltFileOutput:

    """Class for outputing strings in appropriate encoding."""

    def __init__(self, encoding = DEFAULT_LANG, \
                     ofile = DEFAULT_OUTPUT, flush = False):
        """Create an instance of AltFileOutput."""
        self.encoding = encoding
        self.flush    = flush
        self.ofile    = ofile

    def fprint(self, *ostrings):
        """Encode ostrings and print them, flushing the output if necessary.

        If you won't to redirect fprint's output, you will have to re-set
        self.ofile first. Unfortunately, it's not possible to use argument
        syntax like this: *ostrings, ofile = DEFAULT_OUTPUT
        """
        for ostring in ostrings:
            if isinstance(ostring, unicode):
                ostring = ostring.encode(self.encoding)
            print >> self.ofile, ostring,
        print >> self.ofile
        if self.flush:
            self.ofile.flush()
