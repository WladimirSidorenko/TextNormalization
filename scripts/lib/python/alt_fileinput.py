#!/usr/bin/env python2.7

'''A light-weight alternative to standard fileinput library.'''

##################################################################
# Loaded Modules
import sys
import codecs
from fileinput import *
from os import getenv

##################################################################
# Interface
__all__ = ['AltFileInput']

##################################################################
# Constants
DEFAULT_INPUT = sys.stdin

##################################################################
# Class
class AltFileInput:
    '''Class for input and appropriate decoding of input strings.'''
    def __init__(self, *ifiles, **kwargs):
        '''Create an instance of AltFileInput.'''
        # set up input encoding - use environment variable
        # SOCMEDIA_LANG or 'utf-8' by default
        self.encoding = kwargs.get('encoding', getenv('SOCMEDIA_LANG', 'utf-8'))
        # specify how to handle characters, which couldn't be decoded
        self.errors   = kwargs.get('errors', 'strict')
        # allow ifiles to appear both as list and as a kw argument
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
        '''Yield next line or stop iteration if input exhausted.'''
        self.line = self.current_file.readline()
        if self.line == '':
            self.__next_file_()
            self.next()
        self.line = self.line.decode(encoding = self.encoding, \
                                         errors = self.errors).strip()
        self.fnr +=1
        self.nr  +=1
        return self.line

    def __iter__(self):
        '''Standard method for iterator protocol.'''
        return self

    def __stop__(self):
        '''Unconditionally raise StopIteration() error.'''
        raise StopIteration

    def __next_file_(self):
        '''Switch to new file if possible and update counters.'''
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
            raise StopIteration

    def __open__(self, ifile):
        '''Determine type of ifile argument and open it appropriately.'''
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
