##! RE_OPTIONS: re.UNICODE
# -*- coding: utf-8; -*-

# unconditionally split on `!', `?', and `.' followed by (`!'|`?')
([?!](?:[?!.])*|[.](?:[?!.])+)
# be cautious with `.' - check the right context
(\.)\s*(?:[#@A-ZÄÖÜ][a-zäöüß]|$)

##! RE_OPTIONS: re.UNICODE | re.IGNORECASE
# consider retweets as sentence boundaries
((?:RT\s*)?@[\w]+:?)
