#!/usr/bin/env python2.7

##################################################################
# Import Libraries
import ast
import os
import re
import shlex
import sys
from ipopen import IPopen

from comparator import Comparator
from ext_comparator import ExternalComparator

##################################################################
# Constants
SPACES_RE = re.compile(r'[ \t\v\f]+')
EOL_RE    = re.compile(r'[ \t\v\f]?(\r?\n)+[  \t\r\v\f]?')

##################################################################
# Class
class Tester:

    '''Main class of test utility.

    This class sets up a test utility and incorporates all the
    necessary data structures and methods used for testing.

    '''

    def __init__(self, command, command_args = '', \
                     skip_line = '\n\n', \
                     skip_line_expect = '\n', \
                     timeout = 2, name = 'unknown', \
                     strip_spaces = 'True', cmp_command = '', \
                     cmp_opts = '{"func":"_cmp", "ignore_case":False}'):
        '''Create new Tester instance.

        This method establishes a pipe to the command to be tested and
        sets up all the necessary utilities and methods used to
        compare command's output against gold.

        '''
        # setting up processor for input data and all the necessary
        # attributes
        self.skip_line = skip_line
        self.skip_line_expect = skip_line_expect
        self.timeout = int(timeout)
        self.processor = IPopen(args = [command] + shlex.split(command_args.format(**os.environ)), \
                                    skip_line = self.skip_line, \
                                    skip_line_expect = self.skip_line_expect, \
                                    timeout = self.timeout)
        # registering opened file descriptors
        self._open_fds = [self.processor]
        # setting up setname
        self.name = name
        # Deciding whether leading and trailing spaces from testcase
        # gold and input should be stripped. Strip them by default.
        if ast.literal_eval(strip_spaces):
            self.space_handler = self._strip_spaces
        else:
            self.space_handler = self._keep_spaces
        # self._cmp will hold an adress of a function which will be
        # either an external command involved through a pipe or a
        # pythonic function from the class Tester.Comparator. But in
        # both cases function stored under self._cmp should provide
        # same API.
        if cmp_command:
            self._cmp = self._cmp_cmd_wrapper(cmp_command)
        else:
            self._cmp = self._cmp_func_wrapper(cmp_opts)
        # zeroing statistics
        self.s = self.f = 0

    def process(self, _input, encd='utf-8'):
        '''Pass input to processor and return its output.'''
        _output = self.processor.communicate(_input, encd)
        return self.space_handler(_output)

    def cmp(self, gold, output, matcht = "full", \
                be_quiet = False, update_ts_stat = True):
        '''Compare 2 input strings and update statistics if necessary.'''
        if matcht == "full":
            state = (self._cmp(gold, output) == 0) # boolean will be stored
        elif matcht == "included":
            state = (gold in output)
        else:
            raise RuntimeError("Invalid match type specified for test case: {}".format(matcht))
        if state:
            if not be_quiet:
                sys.stderr.write('.')
            if update_ts_stat:
                self.s += 1
        else:
            if not be_quiet:
                sys.stderr.write('F')
            if update_ts_stat:
                self.f += 1
        return state

    def make_report(self):
        '''Generate statistics report on current testset.'''
        total = self.s + self.f
        if total:
            s_perc = self.s / float(total)
            f_perc = 1 - s_perc
        else:
            s_perc = f_perc = 0
        return '''

Statistics Report
Testset:  '{self.name}'
{succ:15s}{self.s:d} ({s_perc:.2%})
{fail:15s}{self.f:d} ({f_perc:.2%})
============================
{tot:15s}{total:d}
'''.format(succ='Succeeded:', fail='Failed:', tot='Total:', **locals())

    def close(self):
        '''Take clean-up actions when Tester instance finishes.'''
        for fd in self._open_fds:
            fd.close()
        return None

    def _strip_spaces(self, _input):
        '''Remove leading, trailing, and contiguous white space from input.'''
        _input = SPACES_RE.sub(' ', _input) # squeeze contiguous whitespaces
        _input = EOL_RE.sub(r'\n', _input) # squeeze contiguous newlines
        return _input.strip()

    def _keep_spaces(self, _input):
        '''Return original input string unmodified.'''
        return _input

    def _cmp_cmd_wrapper(self, cmp_command):
        '''Establish proper pipe connection to external comparator.

        NOT TESTED'''
        ecomparator = ExternalComparator(cmp_command)
        self._open_fds.append(ecomparator)
        return ecomparator.cmp

    def _cmp_func_wrapper(self, cmp_opts=''):
        '''Return function member of class Tester.Comparator.'''
        return Comparator(**ast.literal_eval(cmp_opts)).cmp
