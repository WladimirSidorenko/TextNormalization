##################################################################
# -*- coding: utf-8-unix; comment-start: "# "; -*-
##################################################################
# This file contains regular expressions, which describe occurrences
# of misspelled umlauts (ae, oe, ue) where they should be restored to
# regular form (ä,ö,ü).
# See the correspondings of misspelled umlauts to normal form in
# ./misspelled2umlaut.map
#################################################################

##################################################################
##! RE_OPTIONS: re.UNICODE
# UE
(?<=[^@AEQÄaeqä])([Uu][Ee])(?![Ii])
(?<=\b)([Uu][Ee])(?![Ii])

##################################################################
##! RE_OPTIONS: re.UNICODE
# AE and OE
(?<=[^@AEae])([AaOo][Ee])
(?<=\b)([AaOo][Ee])
(?<=ge)([AaOo][Ee])
