#!/usr/bin/env python
# -*- coding: utf-8 -*-


class AdvClassifier(object):
    def __call__(self, conn, arg_finder):
        # default strategy - to be classified in the future
        return 'inter_sent'
