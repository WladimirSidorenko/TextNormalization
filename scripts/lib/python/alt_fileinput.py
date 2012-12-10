#!/usr/bin/env python2.7

'''A light-weight alternative to standard fileinput library.'''

##################################################################
# Import modules
import sys as __sys__
import codecs as __codecs__
from fileinput import *

##################################################################
# Declare interface
__all__ = ['AltFileInput']

##################################################################
# Class
class AltFileInput:
    ''' '''
    def __init__(self, ifiles = [], encd = 'utf-8'):
        ''' '''
        # Unfortunately we can't perform a default assignment to __sys__.stdin
        # in argument specification, since a user may pass an empty container,
        # which still has to be populated
        if not ifiles:
            ifiles = [__sys__.stdin]
        self.files = ifiles
        self.fcnt  = -1
        self.current_file = None
        self.filename = None
        self.fnr = 0
        self.nr = 0
        self.encd = encd
        self.line = ''
        self.__next_file_()

    def next(self):
        '''Yield next line or stop iteration if input exhausted.'''
        self.line = self.current_file.readline()
        if self.line == '':
            self.__next_file_()
            self.next()
        self.line = self.line.decode(self.encd).strip()
        self.fnr +=1
        self.nr  +=1
        return self.line

    def __iter__(self):
        '''Standrad method for iterator protocol.'''
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
            self.current_file = self.__open__(self.files[self.fcnt], \
                                                  self.encd)
            self.filename = self.current_file.name
            self.fnr = 0
            self.line = ''
        else:
            # if we have exhausted the list of available files, all
            # subsequent calls to self.next will promptly raise
            # StopIterantion error
            self.next = self.__stop__
            raise StopIteration

    def __open__(self, ifile, encd):
        '''Determine type of ifile argument and open it appropriately.'''
        if isinstance(ifile, file):
            # file is already open
            return ifile
        elif isinstance(ifile, str) or isinstance(ifile, buffer):
            if ifile == '-':
                return __sys__.stdin
            # open it otherwise
            return __codecs__.open(ifile, encoding = encd)
        else:
            raise TypeError('Wrong type of argument')
