#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import cpu_count, Process, Queue
from operator import itemgetter
import sys
from time import time
import traceback
import warnings

from eds_segmentation import EDSSegmenter

class Segmenter(object):
    def __init__(self, num_workers=None, observers=None, verbose=True,
                 stream=sys.stdout):
        if num_workers is None:
            num_workers = cpu_count()
        elif num_workers < 1:
            raise ValueError('number of workers must be a positive number')
        if stream is None and verbose:
            stream = sys.stdout
        elif not hasattr(stream, 'write'):
            raise TypeError('stream must have a write method')
        self.num_workers = num_workers
        self.observers = observers or []
        self.verbose = verbose
        self.stream = stream

    def attach(self, observer):
        self.observers.append(observer)
        return self

    def detach(self, observer):
        try:
            self.observers.remove(observer)
        except ValueError:
            pass
        return self

    def segment(self, sents, with_retval=True, init_kwargs={}):
        sent_queue = Queue()
        for sent_no, sent in enumerate(sents):
            sent_queue.put((sent_no, sent))
        sds_queue = Queue()
        self._notify('start')
        self._debug('creating {0} workers'.format(self.num_workers))
        for worker_id in xrange(self.num_workers):
            worker = SegmenterWorker(worker_id, init_kwargs, sent_queue,
                                     sds_queue, self.verbose, self.stream)
            worker.start()
        sds_buffer = []
        for sent_no in xrange(len(sents)):
            sent_no, sds = sds_queue.get()
            if sds is not None:
                self._notify('sds', sent_no, sds)
            if with_retval:
                sds_buffer.append((sent_no, sds))
            else:
                del sds
        for _ in xrange(self.num_workers):
            sent_queue.put(None)
        self._notify('finish')
        if with_retval:
            sds_buffer.sort(key=itemgetter(0))
            return [sds for sent_no, sds in sds_buffer]

    def _notify(self, event, *args, **kwargs):
        for observer in self.observers:
            try:
                callback = getattr(observer, 'on_{0}'.format(event))
            except AttributeError:
                continue
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                if self.verbose:
                    self._warn('Exception in callback {0}.on_{1}: {2}'.format(
                        observer.__class__.__name__, event, exc))

    def _debug(self, msg):
        if self.verbose:
            self.stream.write('DEBUG: {0}\n'.format(msg))

    def _warn(self, msg):
        warnings.warn(msg, RuntimeWarning)


class SegmenterWorker(Process):
    def __init__(self, worker_id, init_kwargs, sent_queue, sds_queue, verbose,
                 stream):
        Process.__init__(self)
        self.worker_id = worker_id
        self.sent_queue = sent_queue
        self.sds_queue = sds_queue
        self.verbose = verbose
        self.stream = stream
        self._segmenter = EDSSegmenter(**init_kwargs)

    def run(self):
        for sent_no, sent in iter(self.sent_queue.get, None):
            start_time = time()
            self._debug('processing sentence {0}'.format(sent_no))
            try:
                sds = self._segmenter.segment(sent)
            except:
                sds = None
                self._warn('Exception raised during segmentation of sentence '
                           '{0}: {1}'.format(sent_no, traceback.format_exc()))
            self.sds_queue.put((sent_no, sds))
            self._debug('processed sentence {0} in {1:.4f} sec'.format(
                sent_no, time() - start_time))

    def _debug(self, msg):
        if self.verbose:
            self.stream.write('DEBUG({0}): {1}\n'.format(self.worker_id, msg))

    def _warn(self, msg):
        warnings.warn(msg, RuntimeWarning)
