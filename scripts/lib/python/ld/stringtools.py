#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
import re
import sys
from datetime import datetime

##################################################################
# Constants
NWORD = 0
LOWER = 1
UPPER = 2
XML_TAG = re.compile(r"(?:\s*<[^<>]+>)+$")
XML_TAG_NAME  = re.compile(r"(?:\s*<\s*/?\s*([^\s/>]*))")
XML_TAG_ATTRS = re.compile(r"""([^=\s]+)=(["'])(.*?)\2""")
TIMEFMT = r"%a %b %d %X +0000 %Y"

##################################################################
# Methods
def adjust_case(str1, str2):
    """Adjust case of characters in str1 to those in str2.

    In case when str1 is longer than str2 the remaining characters
    in str1 will get the case of the last character in str2."""
    ostring = ''
    case1 = case2 = case_diff = 0
    for (i, (c1, c2)) in enumerate(zip(str1, str2)):
        case1 = check_case(c1)
        case2 = check_case(c2)
        if case1:
            case_diff = case2 and (case1 != case2)
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

def check_case(char):
    """Return case of input character."""
    if char.islower():
        return LOWER
    elif char.isupper():
        return UPPER
    else:
        return NWORD

def upcase_capitalize(str1, str2):
    """Change case of str1 depending on str2."""
    # if the 1-st character of 2-nd string is uppercased
    if str2 and str2[0].isupper():
        # check if the rest of the 2-nd string is all uppercased and uppercase
        # the 1-str string too if yes
        if str2[1:].isupper():
            # upcase all the words except special ones
            return __upcase(str1)
        # if the 2-nd string is capitalized, capitalize the 1-st word of the
        # 1-str string tool
        elif str2[1:].islower():
            return __capitalize(str1)
    # otherwise, return str1 unmodified
    return str1

def is_xml_tag(istr):
    """Check if istr as a whole is an XML tag, return bool."""
    return bool(XML_TAG.match(istr))

def parse_xml_line(iline):
    """Analyze iline as if it were a single XML tag.

    Returns a tuple with tag name as first element and dict of attributes as
    second if line could be parsed or None otherwise.
    """
    name  = ""
    attrs = {}
    if not is_xml_tag(iline):
        return None
    else:
        name = XML_TAG_NAME.match(iline).group(1).lower()
        for attrname, delim, value in XML_TAG_ATTRS.findall(iline):
            attrs[attrname] = value
    if name:
        return (name, attrs)
    else:
        return None

def str2time(istr):
    """Convert string representing time to a time object.

    Input to this method is a string of format:
    Mon Jan 16 14:39:00 +0000 2012
    Return value is a datetime object.
    """
    return datetime.strptime(istr, TIMEFMT)

def __upcase(istr):
    """Upcase all tokens in istr but the special ones."""
    return ' '.join([word.upper() if word.isalpha() else word \
                         for word in istr.split()])

def __capitalize(istr):
    """Capitalize all tokens in istr but the special ones."""
    return istr[0].upper() + istr[1:]
