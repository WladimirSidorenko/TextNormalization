#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_index(token_or_sent):
    if hasattr(token_or_sent, 'pid'):
        pid = token_or_sent.pid
    else:
        pid = token_or_sent
    if pid.startswith('s_'):
        pid = pid[2:]
    elif pid.startswith('tok_'):
        pid = pid[4:]
    else:
        assert False, 'unexpected Paula ID {0}'.format(pid)
    return int(pid) - 1
