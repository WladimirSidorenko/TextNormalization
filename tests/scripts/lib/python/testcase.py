#!/usr/bin/env python2.7

##################################################################
# Class
class TestCase:
    '''Typical data structure holding a testcase.'''
    def __init__(self, _input, _gold, \
                     # `space_handler' should be a function to
                 # process contiguous leading and trailing
                 # spaces in input.
                     space_handler = (lambda _str: _str), \
                     reqstate = 'Success', id = 'unknown', \
                     match = "full"):
        '''Create an instance of TestCase.'''
        self.input  = space_handler(_input)
        self.gold   = space_handler(_gold)
        self.output = ''
        self.id     = id
        self.reqstate = True if reqstate == 'Success' else False
        # current state of testcase
        self.state  = False
        self.match  = match
