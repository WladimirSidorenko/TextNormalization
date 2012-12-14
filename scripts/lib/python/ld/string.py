#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

NWORD = 0
LOWER = 1
UPPER = 2

def adjust_case(self, str1, str2):
    '''Adjust case of characters in str1 to those in str2.

    In case when str1 is longer than str2 the remaining characters
    in str1 will get the case of the last character in str2.'''
    ostring = ''
    case1 = case2 = case_diff = 0
    for (i, (c1, c2)) in enumerate(zip(str1, str2)):
        case1 = self.check_case(c1)
        case2 = self.check_case(c2)
        case_diff = case1 and case2 and (case1 != case2)
        # if cases aren't equal and both characters are letters, swap case
        # of the character from the first string
        if case_diff:
            c1 = c1.swapcase()
        ostring += c1
    # append the rest of the 1-st string to output and adjust case if
    # necessary
    i += 1                  # after i-th character comes (i+1)-th boundary
    if case_diff:
        ostring += str1[i:].swapcase()
    else:
        ostring += str1[i:]
    return ostring

def check_case(self, char):
    '''Return case of input character.'''
    if char.islower():
        return self.LOWER
    elif char.isupper():
        return self.UPPER
    else:
        return self.NWORD
