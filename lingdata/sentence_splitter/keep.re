# -*- coding: utf-8; -*-

##################################################################
##! RE_OPTIONS: re.UNICODE
\b((?:\w+[bg]t?l|[^AEIUOYÄÖÜaeiuoyäöü\W\s]+|\w)\.)

##################################################################
##! RE_OPTIONS: re.UNICODE | re.IGNORECASE

# exclude Internet domain names
(\.(?:a(?:ero|sia|[c-gil-oq-uwxz])|b(?:iz|[abd-jmnorstvwyz])|c(?:at|o(?:m|op)|[acdf-ik-orsuvxyz])|d[dejkmoz]|e(?:du|[ceghtu])|f[ijkmor]|g(?:ov|[abd-ilmnp-uwy])|h[kmnrtu]|i(?:n(?:fo|t)|[del-oq-t])|j(?:obs|[emop])|k[eghimnprwyz]|l[abcikr-vy]|m(?:il|obi|useum|[acdeghk-z])|n(?:ame|et|[acefgilopruz])|o(?:m|rg)|p(?:ro|[ae-hk-nrstwy])|qa|r[eosuw]|s[a-eg-or-vxyz]|t(?:(?:rav)?el|[cdfghj-pr])|xxx))\b
