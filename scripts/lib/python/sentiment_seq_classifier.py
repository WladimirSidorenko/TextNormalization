#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for NN-based sentiment classification.

Attributes:

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from alt_fio import AltFileInput

from collections import Counter
from datetime import datetime
from lasagne.init import HeUniform, Orthogonal
from sklearn.model_selection import train_test_split
from theano import config, tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import theano
import sys


##################################################################
# Variables and Constants
def floatX(a_data, a_dtype=config.floatX):
    """Return numpy array populated with the given data.

    Args:
      data (np.array):
        input tensor
      dtype (class):
        digit type

    Returns:
      np.array:
        array populated with the given data

    """
    return np.asarray(a_data, dtype=a_dtype)


np.random.seed()

GRU = "gru"
LSTM = "lstm"

SEQ = "seq"
TREE = "tree"

DFLT_ORDER = 1
DFLT_VSIZE = 100
DFLT_INTM_DIM = 100

INF = float('inf')

UNK = "%UNK%"
UNK_I = 0
UNK_PROB = lambda: np.random.binomial(1, 0.05)

_ORTHOGONAL = Orthogonal()
ORTHOGONAL = lambda x: floatX(_ORTHOGONAL.sample(x))

_HE_UNIFORM = HeUniform()
HE_UNIFORM = lambda x: floatX(_HE_UNIFORM.sample(x))

MAX_ITERS = 150
CONV_EPS = 1e-5

TRUNCATE_GRADIENT = 20
TRNG = RandomStreams()


##################################################################
# Methods
def rmsprop(tparams, grads, x, y, cost):
    """A variant of SGD that automatically scales the step size.

    Args:
      tpramas (Theano SharedVariable):
          Model parameters
      grads (Theano variable):
          Gradients of cost w.r.t to parameres
      x (list):
          Model inputs
      y (Theano variable):
          Targets
      cost (Theano variable):
          Objective fucntion to minimize

    Notes:
      For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    """
    zipped_grads = [theano.shared(p.get_value() * floatX(0.))
                    for p in tparams]
    running_grads = [theano.shared(p.get_value() * floatX(0.))
                     for p in tparams]
    running_grads2 = [theano.shared(p.get_value() * floatX(0.))
                      for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(x + [y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')
    updir = [theano.shared(p.get_value() * floatX(0.))
             for p in tparams]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / TT.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams, updir_new)]
    f_update = theano.function([], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')
    params = [zipped_grads, running_grads, running_grads2, updir]
    return (f_grad_shared, f_update, params)


##################################################################
# Class
class SentimenSeqClassifier(object):
    """Class for neural net-based sentiment classification.

    """

    def __init__(self, a_w2v, a_type=GRU, a_topology=SEQ,
                 a_order=DFLT_ORDER):
        """Class constructor.

        Args:
          a_w2v (file path or None):
            use pre-trained word2vec instance
          a_type (GRU or LSTM):
            type of the RNN to construct and train
          a_topology (SEQ or TREE):
            topology of the RNN to construct and train
          a_order (int):
            order of the linear RNN

        """
        self.w2v = bool(a_w2v)
        self._w2v_path = a_w2v
        self._type = a_type
        self._topology = a_topology
        self._order = a_order
        # conversion from symbolic form to feature indices
        self._x2idx = {}
        self._y2idx = {}
        self._params = []
        self.W_INDICES = self.W_EMB = None
        self.Y_pred = self.Y_gold = None
        self.w_i = self.n_y = self.ndim = self.intm_dim = -1
        self.use_dropout = theano.shared(floatX(0.))
        self._grad_shared = self._update = self._rms_params = None
        # methods
        self._predict_labels = self._predict_probs = self._debug_nn = None
        self._compute_acc = self._compute_cross_ent = None

    def train(self, X_train, Y_train, X_dev=[], Y_dev=[]):
        """Construct and train a neural network.

        Args:
          X_train (list):
            training instances as words
          Y_train (list):
            gold labels as symbolic tags
          X_dev (list):
            dev instances as words
          Y_dev (list):
            gold labels as symbolic tags

        """
        if not X_dev:
            X_train, X_dev, Y_train, Y_dev = train_test_split(
                X_train, Y_train, test_size=0.1)
        # load embeddings
        self._x2idx = {UNK: UNK_I}
        xset = set(x for X in (X_train, X_dev)
                   for x_inst in X
                   for x in x_inst)
        self.w_i = len(xset) + 1
        if self.w2v:
            self._load_emb(self._w2v_path, lambda w: w in xset)
        else:
            self.ndim = DFLT_VSIZE
            self.W_EMB = theano.shared(
                value=HE_UNIFORM((self.w_i, self.ndim)), name="W_EMB")
        # digitize input
        X_train = self._digitize_X(X_train, not self.w2v)
        X_dev = self._digitize_X(X_dev, False)
        # digitize labels
        Y_train = self._digitize_Y(Y_train, True)
        Y_dev = self._digitize_Y(Y_dev, False)
        # initialize network
        self.use_dropout.set_value(1.)
        self._init_nnet()
        grads = TT.grad(self._cross_ent, wrt=self._params)
        self._init_funcs(grads)
        # train
        self._train(X_train, Y_train, X_dev, Y_dev)
        # switch-off dropout
        self.use_dropout.set_value(0.)

    def _train(self, X_train, Y_train, X_dev, Y_dev):
        """Perform the actual training.

        Args:
          X_train (list[np.array]): training instances as embedding indices
          Y_train (list[np.array]): gold labels for the training instances
          X_dev (list[np.array]): dev set instances as embedding indices
          Y_dev (list[np.array]): gold labels for the dev set instances

        Returns:
          void:

        """
        n_train = float(len([x for x_inst in X_train for x in x_inst]))
        n_dev = float(len([x for x_inst in X_dev for x in x_inst]))
        train_ce = dev_acc = dev_ce = 0.
        max_dev_acc = dev_acc = -INF
        prev_train_ce = train_ce = min_dev_ce = dev_ce = INF
        start_time = end_time = time_delta = None
        best_params = []
        for i in xrange(MAX_ITERS):
            start_time = datetime.utcnow()
            # perform one training iteration
            train_ce = 0.
            for x, (y_lbl, y_prob) in zip(X_train, Y_train):
                train_ce += self._grad_shared(x, y_prob)
                self._update()
            train_ce /= n_train
            # temporarily deactivate dropout
            self.use_dropout.set_value(0.)
            # estimate the model on the dev set
            dev_acc = dev_ce = 0.
            for x, (y_lbl, y_prob) in zip(X_dev, Y_dev):
                # print("x =", repr(x), file=sys.stderr)
                # print("y_lbl =", repr(y_lbl), file=sys.stderr)
                # print("y_pred_lbl =", repr(self._predict_labels(x)),
                #       file=sys.stderr)
                # print("acc =", repr(self._compute_acc(x, y_lbl)),
                #       file=sys.stderr)
                # print("y_prob =", repr(y_prob), file=sys.stderr)
                # print("y_pred_prob =", repr(self._predict_probs(x)),
                #       file=sys.stderr)
                # print("ce =", repr(self._compute_cross_ent(x, y_prob)),
                #       file=sys.stderr)
                dev_acc += self._compute_acc(x, y_lbl)
                dev_ce += self._compute_cross_ent(x, y_prob)
            dev_acc /= n_dev
            dev_ce /= n_dev
            # switch dropout on again
            self.use_dropout.set_value(1.)
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).seconds
            if dev_acc > max_dev_acc or \
               (dev_acc == max_dev_acc and dev_ce < min_dev_ce):
                best_params = [p.get_value() for p in self._params]
                max_dev_acc = dev_acc
                min_dev_ce = dev_ce
            print("Iteration {:d}:\ttrain_ce = {:f}\t"
                  "dev_acc={:f}\tdev_ce={:f}\t({:.2f} sec)".format(
                      i, train_ce, dev_acc, dev_ce, time_delta),
                  file=sys.stderr)
            if abs(prev_train_ce - train_ce) < CONV_EPS:
                break
            prev_train_ce = train_ce
        # deactivate dropout
        if best_params:
            for p, val in zip(self._params, best_params):
                p.set_value(val)
        else:
            raise RuntimeError("Network could not be trained.")

    def save(self, a_path):
        """Save neural network to disc.

        Args:
          a_path (str):
            path for toring the model

        """
        pass

    def _digitize_X(self, a_X, a_train=False):
        """Convert symbolic y labels to numpy arrays.

        Args:
          a_X (list):
            symbolic input
          a_train (bool):
            create internal mapping from symbolic input to indices

        Returns:
          (list): list of binary vectors

        """
        ret_X = []
        if a_train:
            new_x_inst = None
            x_stat = Counter(x
                             for x_inst in a_X
                             for x in x_inst)
            self._x2idx = {UNK: UNK_I}
            for x_inst in a_X:
                new_x_inst = np.empty((len(x_inst),), dtype="int32")
                for i, x in enumerate(x_inst):
                    if x in self._x2idx:
                        new_x_inst[i] = self._x2idx[x]
                    elif x_stat[x] > 1 or UNK_PROB():
                        new_x_inst[i] = self._x2idx[x] = \
                            len(self._x2idx)
                    else:
                        new_x_inst[i] = UNK_I
                ret_X.append(new_x_inst)
        else:
            for x_inst in a_X:
                ret_X.append([self._x2idx.get(x, UNK_I)
                              for x in x_inst])
        return ret_X

    def _digitize_Y(self, a_Y, a_train=False):
        """Convert symbolic y labels to numpy arrays.

        Args:
          a_Y (list):
            symbolic gold labels
          a_train (bool):
            create internal mapping from symbolic labels to indices

        Returns:
          (list): list of binary vectors

        """
        ret_Y = []
        if a_train:
            for y_inst in a_Y:
                for y in y_inst:
                    if y not in self._y2idx:
                        self._y2idx[y] = len(self._y2idx)
                        self.n_y
            self.n_y = len(self._y2idx)
        y_lbl = y_prob = None
        for y_inst in a_Y:
            y_lbl = np.zeros((1, len(y_inst)), dtype="int32")
            y_prob = np.zeros((len(y_inst), self.n_y))
            for i, y in enumerate(y_inst):
                y_lbl[0, i] = self._y2idx[y]
                y_prob[i, self._y2idx[y]] = 1.
            ret_Y.append((y_lbl, floatX(y_prob)))
        return ret_Y

    def _load_emb(self, a_path, a_remember_word=lambda w: True):
        """Load pre-trained word embeddings.

        Args:
          a_path (str):
            path to pretrained word embeddings
          a_remember_word (lambda):
            custom function for checking which words to retrieve

        Returns:
          (void):

        Note:
          populates an internal dictionary

        """
        # iterate over name of polarity dictionaries
        W_EMB = None
        word = ""
        fields = []
        first_line = True
        finput = AltFileInput(a_path)
        for iline in finput:
            if not iline:
                continue
            elif first_line:
                _, self.ndim = iline.split()
                W_EMB = np.empty((self.w_i, self.vsize))
                W_EMB[UNK_I, :] = 1e-2
                first_line = False
                continue
            fields = iline.split()
            word = fields[0]
            if a_remember_word(word):
                W_EMB[len(self._x2idx), :] = [float(v) for v in fields[1:]]
                self._x2idx[word] = len(self._x2idx)
        self.W_EMB = theano.shared(value=W_EMB, name="W_EMB")

    def _init_nnet(self):
        """Initialize neural network.

        Args:

        Returns:
          void:

        """
        self.intm_dim = max(100, self.ndim - (self.ndim - self.n_y) / 2)
        self.W_INDICES = TT.ivector(name="W_INDICES")
        self.EMB = self.W_EMB[self.W_INDICES]
        invars = ((self.EMB, False),)
        if self._topology == SEQ and self._order > 1:
            self.intm_dim *= self._order
        if self._type == GRU:
            params, outvars = self._init_gru(invars)
        elif self._type == LSTM:
            params, outvars = self._init_lstm(invars)
        else:
            raise NotImplementedError
        self._params.extend(params)
        self.RNN = outvars[0]
        self.RNN2OUT = theano.shared(
            value=HE_UNIFORM((self.intm_dim, self.n_y)),
            name="RNN2OUT")
        self._params.append(self.RNN2OUT)
        self.OUT = TT.dot(self.RNN, self.RNN2OUT)
        self.Y_pred_probs = TT.nnet.softmax(self.OUT)
        self.Y_pred_labels = TT.argmax(self.OUT, axis=1)
        self.Y_gold_probs = TT.matrix(name="Y_gold_probs")
        self.Y_gold_labels = TT.matrix(name="Y_gold_labels")
        # use accuracy as primary cost function
        self._acc = TT.sum(TT.eq(self.Y_pred_labels, self.Y_gold_labels))
        # use cross-entropy as secondary cost function
        # (cf. https://en.wikipedia.org/wiki/Cross_entropy)
        self._cross_ent = -TT.sum(
            self.Y_gold_probs * TT.log(self.Y_pred_probs)
            + (TT.ones_like(self.Y_gold_probs) - self.Y_gold_probs)
            * TT.log(TT.ones_like(self.Y_pred_probs) - self.Y_pred_probs))

    def _init_gru(self, a_invars, a_sfx="-forward"):
        """Initialize LSTM layer.

        Args:
          a_invars (list[theano.shared]):
              list of input parameters as symbolic theano variable
          a_sfx (str):
            suffix to use for function and parameter names

        Returns:
          (2-tuple):
            parameters to be optimized and list of symbolic outputs from the
            function

        """
        intm_dim = self.intm_dim
        ones = TT.ones((intm_dim,), dtype=config.floatX)
        # initialize transformation matrices and bias term
        W_dim = (intm_dim, self.ndim)
        W = np.concatenate([ORTHOGONAL(W_dim), ORTHOGONAL(W_dim),
                            ORTHOGONAL(W_dim)], axis=0)
        W = theano.shared(value=W, name="W" + a_sfx)
        # bias vector for `W`
        b_dim = (1, intm_dim * 3)
        b = theano.shared(value=HE_UNIFORM(b_dim), name="b" + a_sfx)

        U_dim = (intm_dim, intm_dim)
        U_rz = np.concatenate([ORTHOGONAL(U_dim), ORTHOGONAL(U_dim)], axis=0)
        U_rz = theano.shared(value=U_rz, name="U_rz" + a_sfx)

        U_h = theano.shared(value=ORTHOGONAL(U_dim), name="U_h" + a_sfx)

        params = [W, U_rz, U_h, b]

        # initialize dropout units
        w_do = theano.shared(value=floatX(np.ones((3 * intm_dim,))),
                             name="w_do")
        w_do = self._init_dropout(w_do)
        u_rz_do = theano.shared(value=floatX(np.ones((2 * intm_dim,))),
                                name="u_rz_do")
        u_rz_do = self._init_dropout(u_rz_do)
        u_h_do = theano.shared(value=floatX(np.ones((intm_dim,))),
                               name="u_h_do")
        u_h_do = self._init_dropout(u_h_do)

        # custom function for splitting up matrix parts
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        # define recurrent GRU unit
        def _step(x_, h_, W, U_rz, U_h,
                  b, w_do, u_rz_do, u_h_do):
            """Recurrent LSTM unit.

            Note:
              The general order of function parameters to fn is: sequences (if
              any), prior result(s) (if needed), non-sequences (if any)

            Args:
              x_ (theano.shared): input vector
              o_ (theano.shared): output vector
              h_ (theano.shared): previous hidden vector
              W (theano.shared): input transform matrix
              U_rz (theano.shared): inner-state transform matrix for reset and
                update gates
              U_h (theano.shared): inner-state transform matrix for the
                hidden state
              b (theano.shared): bias vector
              w_do (TT.col): dropout unit for the W matrix
              u_rz_do (TT.col): dropout unit for the U_rz matrix
              u_h_do (TT.col): dropout unit for the U_h matrix

            Returns:
              2-tuple: (new output and hidden states)

            """
            # pre-compute common terms:
            # W \in R^{300 x 100}
            # x \in R^{1 x 100}
            # U_rz \in R^{200 x 100}
            # U_h \in R^{100 x 100}
            # h \in R^{1 x 59}
            # b \in R^{1 x 300}
            # w_do \in R^{236 x 1}
            # u_do \in R^{236 x 1}
            wb = TT.dot(W * w_do.dimshuffle((0, 'x')), x_.T).T + b
            urz = TT.dot(U_rz * u_rz_do.dimshuffle((0, 'x')),
                         h_.T).dimshuffle(('x', 0))
            # update: z \in R^{1 x 100}
            z = TT.nnet.sigmoid(
                _slice(wb, 0, intm_dim)
                + _slice(urz, 0, intm_dim))
            # reset: r \in R^{1 x 100}
            r = TT.nnet.sigmoid(_slice(wb, 1, intm_dim)
                                + _slice(urz, 1, intm_dim))
            # preliminary activation: h \in R^{1 x 59}
            h_tilde = TT.tanh(_slice(wb, 2, intm_dim)
                              + TT.dot(U_h * u_h_do.dimshuffle((0, 'x')),
                                       (r * h_).T).T)
            h = z * h_ + (ones - z) * h_tilde
            # return current output and memory state
            return h.flatten()

        m = 0
        outvars = []
        for iv, igbw in a_invars:
            m = iv.shape[0]
            ret, _ = theano.scan(_step,
                                 sequences=[iv],
                                 outputs_info=[
                                     floatX(np.zeros((self.intm_dim,)))],
                                 non_sequences=[W, U_rz, U_h, b,
                                                w_do, u_rz_do, u_h_do],
                                 name="GRU" + str(iv) + a_sfx,
                                 n_steps=m,
                                 truncate_gradient=TRUNCATE_GRADIENT,
                                 go_backwards=igbw)
            outvars.append(ret)
        return params, outvars

    def _init_lstm(self, a_invars, a_sfx="-forward"):
        """Initialize LSTM layer.

        Args:
          a_invars (list[theano.shared]):
              list of input parameters as symbolic theano variable
          a_sfx (str):
            suffix to use for function and parameter names

        Returns:
          (2-tuple):
            parameters to be optimized and list of symbolic outputs from the
            function

        """
        intm_dim = self.intm_dim
        # initialize transformation matrices and bias term
        W_dim = (intm_dim, self.ndim)
        W = np.concatenate([ORTHOGONAL(W_dim), ORTHOGONAL(W_dim),
                            ORTHOGONAL(W_dim), ORTHOGONAL(W_dim)],
                           axis=0)
        W = theano.shared(value=W, name="W" + a_sfx)

        U_dim = (intm_dim, intm_dim)
        U = np.concatenate([ORTHOGONAL(U_dim), ORTHOGONAL(U_dim),
                            ORTHOGONAL(U_dim), ORTHOGONAL(U_dim)],
                           axis=0)
        U = theano.shared(value=U, name="U" + a_sfx)

        V = ORTHOGONAL(U_dim)   # V for vendetta
        V = theano.shared(value=V, name="V" + a_sfx)

        b_dim = (1, intm_dim * 4)
        b = theano.shared(value=HE_UNIFORM(b_dim), name="b" + a_sfx)

        params = [W, U, V, b]

        # initialize dropout units
        w_do = theano.shared(value=floatX(np.ones((4 * intm_dim,))),
                             name="w_do")
        w_do = self._init_dropout(w_do)
        u_do = theano.shared(value=floatX(np.ones((4 * intm_dim,))),
                             name="u_do")
        u_do = self._init_dropout(u_do)

        # custom function for splitting up matrix parts
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        # define recurrent LSTM unit
        def _step(x_, h_, c_,
                  W, U, V, b, w_do, u_do):
            """Recurrent LSTM unit.

            Note:
              The general order of function parameters to fn is: sequences (if
              any), prior result(s) (if needed), non-sequences (if any)

            Args:
              x_ (theano.shared): input vector
              h_ (theano.shared): output vector
              c_ (theano.shared): memory state
              W (theano.shared): input transform matrix
              U (theano.shared): inner-state transform matrix
              V (theano.shared): output transform matrix
              b (theano.shared): bias vector
              w_do (TT.col): dropout unit for the W matrix
              u_do (TT.col): dropout unit for the U matrix

            Returns:
              (2-tuple(h, c))
                new hidden and memory states

            """
            # pre-compute common terms:
            # W \in R^{236 x 100}
            # x \in R^{1 x 100}
            # U \in R^{236 x 59}
            # h \in R^{1 x 59}
            # b \in R^{1 x 236}
            # w_do \in R^{236 x 1}
            # u_do \in R^{236 x 1}

            # xhb \in R^{1 x 236}
            xhb = (TT.dot(W * w_do.dimshuffle((0, 'x')), x_.T) +
                   TT.dot(U * u_do.dimshuffle((0, 'x')), h_.T)).T + b
            # i \in R^{1 x 59}
            i = TT.nnet.sigmoid(_slice(xhb, 0, intm_dim))
            # f \in R^{1 x 59}
            f = TT.nnet.sigmoid(_slice(xhb, 1, intm_dim))
            # c \in R^{1 x 59}
            c = TT.tanh(_slice(xhb, 2, intm_dim))
            c = i * c + f * c_
            # V \in R^{59 x 59}
            # o \in R^{1 x 59}
            o = TT.nnet.sigmoid(_slice(xhb, 3, intm_dim) +
                                TT.dot(V, c.T).T)
            # h \in R^{1 x 59}
            h = o * TT.tanh(c)
            # return current output and memory state
            return h.flatten(), c.flatten()

        m = 0
        n = intm_dim
        ov = None
        outvars = []
        for iv, igbw in a_invars:
            m = iv.shape[0]
            ret, _ = theano.scan(_step,
                                 sequences=[iv],
                                 outputs_info=[floatX(np.zeros((n,))),
                                               floatX(np.zeros((n,)))],
                                 non_sequences=[W, U, V, b, w_do, u_do],
                                 name="LSTM" + str(iv) + a_sfx,
                                 n_steps=m,
                                 truncate_gradient=TRUNCATE_GRADIENT,
                                 go_backwards=igbw)
            ov = ret[0]
            outvars.append(ov)
        return params, outvars

    def _init_dropout(self, a_input):
        """Create a dropout layer.

        Args:
          a_input (theano.vector): input layer

        Returns:
          theano.vector: dropout layer

        """
        # the dropout layer itself
        output = TT.switch(self.use_dropout,
                           a_input * (TRNG.binomial(a_input.shape, p=0.5, n=1,
                                                    dtype=a_input.dtype)),
                           a_input * floatX(0.5))
        return output

    def _init_funcs(self, a_grads=None):
        """Compile theano functions.

        Args:
          a_grads (theano.shared or None):
            gradients of the trainign function

        Returns:
          void:

        Note:
          modifies instance variables in place

        """
        if a_grads:
            self._grad_shared, self._update, \
                self._rms_params = rmsprop(self._params, a_grads,
                                           [self.W_INDICES],
                                           self.Y_gold_probs,
                                           self._cross_ent)
        if self._predict_labels is None:
            self._predict_labels = theano.function(
                [self.W_INDICES], self.Y_pred_labels,
                name="_predict_labels"
            )

        if self._predict_probs is None:
            self._predict_probs = theano.function([self.W_INDICES],
                                                  self.Y_pred_probs,
                                                  name="_predict_probs")
        if self._compute_acc is None:
            self._compute_acc = theano.function(
                [self.W_INDICES, self.Y_gold_labels],
                self._acc, name="_compute_acc"
            )

        if self._compute_cross_ent is None:
            self._compute_cross_ent = theano.function(
                [self.W_INDICES, self.Y_gold_probs],
                self._cross_ent, name="_compute_cross_ent"
            )

        # initialize debug function
        if self._debug_nn is None:
            self._debug_nn = theano.function([self.W_INDICES],
                                             [self.Y_pred_probs],
                                             name="_debug_nn")
