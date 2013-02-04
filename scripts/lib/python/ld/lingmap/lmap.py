#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

##################################################################
# Libraries
from alt_fileinput import AltFileInput
from .. import __re__, skip_comments, DEFAULT_RE, RuleFormatError
from .  import MAP_DELIMITER

##################################################################
# Class
class Map:
    '''Class for replacing input text according to loaded rules.'''

    def __init__(self, ifile = None, encd = 'utf8'):
        '''Read map entries from ifile and transform them to dict.

        Map entries will be read from input file ifile which should have the form:
        src_entry \t trg_entry
        These entries will be transformed to a dict of form:
        map[src_entry] = trg_entry
        Additinally a special regular expression will be generated from dict keys.
        '''
        self.map   = {}
        self.encd  = encd
        self.flags = []
        src = trg = ''
        # load entries from ifile if it is specified
        if ifile:
            self.map = self.__load(ifile)
        # initialize instance variables
        self.re = self.__compile_re(self.map, *self.flags)

    def reverse(self, lowercase_key = False):
        '''Return reverse copy of map.'''
        # create an empty object of same class
        ret = self.__class__(None)
        # copy over all entries from self.map and swap key and value
        for src, trg in self.map.items():
            if lowercase_key:
                src = src.lower()
            if (trg in ret.map) and (src != ret.map[trg]):
                raise RuntimeError('''
Could not reverse map. Duplicate translation variants for '{:s}':
{:s} vs. {:s}'''.format(trg.encode('utf-8'), \
                            ret.map[trg].encode('utf-8'), \
                            src.encode('utf-8')))
            ret.map[trg] = src
        ret.re = self.__compile_re(ret.map)
        return ret

    def replace(self, istring):
        '''Replace all occurrences of src entries with trg in ISTRING.

        Search in ISTRING for all occurrences of src map entries and
        replace them with their corresponding trg form from self.map'''
        istring = self.re.sub(lambda m: self.map[m.group(0)], istring)
        return istring

    # self.sub will be an alias for self.reverse
    sub = replace

    ##################
    # Private Methods
    def __load(self, ifile):
        '''Load map entries from file ifile.'''
        # load map entries from file
        output = {}
        finput = AltFileInput(ifile, encd = self.encd)
        for line in finput:
            if line:
                # find map entries
                line = skip_comments(line)
                m = MAP_DELIMITER.search(line)
                if m:
                    src, trg = self.__normalize_quotes(line[0:m.start()], \
                                                           line[m.end():])
                    if not (src and trg):
                        print src.encode('utf-8')
                        print trg.encode('utf-8')
                        raise RuleFormatError(efile = finput)
                    output[src] = trg
                elif line:
                    raise RuleFormatError(efile = finput)
        return output

    def __normalize_quotes(self, *istrings):
        '''Normalize quotes at begin and end of istring.

        Unescaped qutes at the beginning and and of a string are
        stripped and escaped quotes with literal ones E.g.

        "Hello World" --> Hello World
        \"Hello World\" --> "Hello World"

        NOTE: only a single occurrence of unescaped quotes is
        stripped.
        '''
        output = []
        for istring in istrings:
            lngth = len(istring)
            quotes = istring[0] + istring[-1]
            esc_quotes = istring[0:2] + istring[-2:]
            if lngth > 1 and (quotes == "''" or quotes == '""'):
                istring = istring[1:-1]
            elif lngth > 3 and (esc_quotes == r"\'\'" or esc_quotes == r'\"\"'):
                istring = istring[1:-2] + istring[-1:]
            output.append(istring)
        return output

    def __compile_re(self, idict, *flags):
        '''Compile RE from keys of given DICT_OBJ.'''
        if not len(idict):
            return DEFAULT_RE
        return __re__.compile('(?:' + '|'.join([k for k in idict]) + \
                                  ')', *flags)
