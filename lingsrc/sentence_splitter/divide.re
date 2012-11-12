# -*- coding: utf-8; comment-start: "# " -*-

##################################################################
# Macros:

##################################################################
##! RE_OPTIONS: re.UNICODE

# unconditionally split on `!', `?', and `.' followed by (`!'|`?')
([—?!](?:[?!.])*|[.](?:[?!.])+)

# be cautious with `.' - check the right context
([.])\s*(?:[^a-zäöüß\s]|$)

# divide by dash if context is obvious
(\s)-\s*[A-Z](?:\w{,2}|\w+ig)\b

# quotation mark followed by hashtags till eol
\w(['"])(?:\s#\w+)+$

##################################################################
##! RE_OPTIONS: re.UNICODE | re.IGNORECASE

# consider retweets as sentence boundaries
((?:RT\s*)?@[\w]+:?)