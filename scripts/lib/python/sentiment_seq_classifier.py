#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for NN-based sentiment classification.

Attributes:

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from alt_fio import AltFileInput

from collections import Counter, defaultdict
from cPickle import dump, load
from datetime import datetime
from lasagne.init import HeUniform, Orthogonal
from sklearn.model_selection import KFold, train_test_split
from theano import config, tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import os
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

M_TRAIN = 0
M_TEST = 1

UNK = "%UNK%"
UNK_I = 0
UNK_PROB = lambda: np.random.binomial(1, 0.05)

_ORTHOGONAL = Orthogonal()
ORTHOGONAL = lambda x: floatX(_ORTHOGONAL.sample(x))

_HE_UNIFORM = HeUniform()
HE_UNIFORM = lambda x: floatX(_HE_UNIFORM.sample(x))

MAX_ITERS = 256  # 150
START_DROPOUT = MAX_ITERS / 3
CONV_EPS = 1e-5
N_RESAMPLE = MAX_ITERS / 3

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


def _spans2stat(a_labels, a_Y):
    """Compute statistics on labeled spans.

    Args:
      a_labels (list[int]): list of all possible labels
      a_Y (list[list]): list of assigned labels

    Returns:
      3-tuple: mapping from labels to spans, spans to toks, and from toks
        to labels

    """
    lbl2spans = {ilbl: set() for ilbl in a_labels}
    span2toks = defaultdict(set)
    tok2lbl = {}
    tok_cnt = 0
    span_cnt = -1
    prev_lbl = ""
    for isent in a_Y:
        # print("isent =", repr(isent))
        prev_lbl = ""
        for ilbl in isent:
            # print("ilbl =", repr(ilbl))
            if ilbl != prev_lbl:
                span_cnt += 1
                lbl2spans[ilbl].add(span_cnt)
                prev_lbl = ilbl
            tok2lbl[tok_cnt] = ilbl
            span2toks[span_cnt].add(tok_cnt)
            tok_cnt += 1
    return (lbl2spans, span2toks, tok2lbl)


def compute_macro_f1(a_labels, a_gold_stat, a_preds):
    """Compute macro-averaged F1 score.

    Args:
      a_labels (iterator): gold tags
      a_gold (list[tuple]): gold labels
      a_preds (list[tuple]): predicted labels

    Returns:
      float: macro-averaged F1 score for all labels

    """
    ntags = len(a_labels)
    macro_f1 = 0.0
    span_toks = None
    prec = rcall = fscore = 0.0
    gold_lbl2span, gold_span2toks, gold_tok2lbl = a_gold_stat
    auto_lbl2span, auto_span2toks, auto_tok2lbl = \
        _spans2stat(a_labels, a_preds)
    for ilbl in a_labels:
        # print("ilbl =", repr(ilbl))
        gold_span_cnt = len(gold_lbl2span[ilbl])
        auto_span_cnt = len(auto_lbl2span[ilbl])
        # print("gold_span_cnt =", repr(gold_span_cnt))
        # print("auto_span_cnt =", repr(auto_span_cnt))

        prec = rcall = fscore = 0.0
        for ispan in auto_lbl2span[ilbl]:
            span_toks = auto_span2toks[ispan]
            prec += float(
                len(set(itok for itok in span_toks
                        if gold_tok2lbl[itok] == ilbl))) / \
                float(len(span_toks))
        prec /= auto_span_cnt or 1e10

        for ispan in gold_lbl2span[ilbl]:
            span_toks = gold_span2toks[ispan]
            rcall += float(
                len(set(itok for itok in span_toks
                        if auto_tok2lbl[itok] == ilbl))) / \
                float(len(span_toks))

        rcall /= gold_span_cnt or 1e10
        fscore = 2 * prec * rcall / (
            (prec + rcall) if (prec or rcall) else 1e10)
        macro_f1 += fscore
    return macro_f1 / (ntags or 1e10)


##################################################################
# Class
class SentimenSeqClassifier(object):
    """Class for neural net-based sentiment classification.

    """

    @classmethod
    def load(self, a_path, a_w2v_path="", a_wselect_func=None):
        """Load neural network from disc.

        Args:
          a_path (str):
            path for toring the model
          a_w2v_path (str):
            (optional) path for word2vec vectors
          a_wselect_func (lamda or None):
            custom function for selecting which word embeddings should be
            loaded

        """
        with open(a_path, "rb") as ifile:
            model = load(ifile)
        if not model.use_w2v:
            if a_w2v_path:
                print("WARNING: Loaded model does not support custom word"
                      " embeddings.")
        else:
            # if no alternative path to word vectors is provided, load from the
            # original path used during training
            if a_w2v_path:
                model._load_emb(a_w2v_path, a_wselect_func)
                model._w2v_path = a_w2v_path
            else:
                model._load_emb(model._w2v_path, a_wselect_func)
        if model.use_w2v:
            model._digitize_X = model._digitize_X_w2v
            model._predict_labels = model._predict_labels_emb
        return model

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
        self.use_w2v = bool(a_w2v)
        self._w2v_path = a_w2v
        self._w2v = {}

        self._type = a_type
        self._topology = a_topology
        self._order = a_order
        assert self._order > 0, "Invalid order {:s}.".format(self._order)
        assert self._topology != TREE or self._order == 1, \
            "Invalid order specified for tree-structured model."
        # conversion from symbolic form to feature indices
        self._x2idx = {UNK: UNK_I}
        self._y2idx = {}
        self._idx2y = {}
        self._params = []
        self.W_INDICES = self.W_EMB = self.w_emb = None
        self.Y_pred = self.Y_gold = None
        self.w_i = self.n_y = self.ndim = self.intm_dim = -1
        self.use_dropout = theano.shared(floatX(0.))
        self._grad_shared = self._update = self._rms_params = None
        # methods
        self._predict_labels = self._predict_labels_emb = None
        self._predict_probs = self._debug_nn = None
        self._compute_acc = self._compute_loss = None

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
        self._balance_ds(X_train, Y_train)
        # load embeddings
        self._x2idx = {UNK: UNK_I}
        xset = set(x for X in (X_train, X_dev)
                   for x_inst in X
                   for x in x_inst)
        self.w_i = len(xset) + 1
        if self.use_w2v:
            self._load_emb(self._w2v_path,
                           lambda w: w in xset, M_TRAIN)
        else:
            self.ndim = DFLT_VSIZE
            self.W_EMB = theano.shared(
                value=HE_UNIFORM((self.w_i, self.ndim)), name="W_EMB")
            self._params.append(self.W_EMB)
        # digitize input
        X_train = [x for x in self._digitize_X(X_train, True)]
        X_dev = [x for x in self._digitize_X(X_dev, False)]
        # digitize labels
        Y_train = self._digitize_Y(Y_train, True)
        Y_dev = self._digitize_Y(Y_dev, False)
        Y_dev_stat = _spans2stat(self._y2idx.itervalues(),
                                 [lbls[0] for lbls, _ in Y_dev])
        X_train, Y_train = self._shuffle(X_train, Y_train)
        # initialize network
        self._init_nnet()
        grads = TT.grad(self._loss, wrt=self._params)
        self._init_funcs(grads)
        # train
        self._train(X_train, Y_train, X_dev, Y_dev, Y_dev_stat)
        # switch-off dropout
        self.use_dropout.set_value(0.)
        # reset loss and related functions in order to dump the model
        self._compute_loss = self._loss = None
        self._grad_shared = self._update = self._rms_params = None

    def predict(self, a_x, a_w2v=None):
        """Predict.

        Args:
          a_x (list):
            single input instance to be classified
          a_w2v (str or None):
            path to embeddings to be loaded

        Returns:
          list: predicted labels

        """
        # convert input instance to the appropariate format
        for x in self._digitize_X([a_x]):
            return [self._idx2y[i] for i in self._predict_labels(x)]

    def save(self, a_path):
        """Save neural network to disc.

        Args:
          a_path (str):
            path for toring the model

        """
        self._w2v = {}
        dirname = os.path.dirname(a_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        elif not os.path.exists(a_path):
            if not os.access(dirname, os.W_OK) or \
               not os.path.isdir(dirname):
                raise RuntimeError("Cannot write to directory '{:s}'.".format(
                    dirname))
        else:
            if not os.access(a_path, os.W_OK):
                raise RuntimeError("Cannot write to file '{:s}'.".format(
                    a_path))
        with open(a_path, "wb") as ofile:
            dump(self, ofile)

    def _digitize_X(self, a_X, a_train=False):
        """Convert symbolic y labels to numpy arrays.

        Args:
          a_X (list):
            symbolic input
          a_train (bool):
            create internal mapping from symbolic input to indices

        Yields:
          (list): digitized input vector

        """
        if a_train:
            new_x_inst = None
            x_stat = Counter(x
                             for x_inst in a_X
                             for x in x_inst)
            for x_inst in a_X:
                new_x_inst = np.empty((len(x_inst),), dtype="int32")
                for i, x in enumerate(x_inst):
                    if x in self._x2idx:
                        new_x_inst[i] = self._x2idx[x]
                    elif x_stat[x] < 2 and UNK_PROB():
                        new_x_inst[i] = UNK_I
                    else:
                        new_x_inst[i] = self._x2idx[x] = \
                            len(self._x2idx)
                yield new_x_inst
        else:
            for x_inst in a_X:
                yield [self._x2idx.get(x, UNK_I) for x in x_inst]

    def _digitize_X_w2v(self, a_X):
        """Convert input to matrix of pre-trained embeddings.

        Args:
          a_X (list): input instances

        Returns:
          np.array: embedding matrices for the input instances

        """
        ret = []
        for x_inst in a_X:
            x_emb = floatX(np.empty((len(x_inst), self.ndim)))
            for i, x in enumerate(x_inst):
                if x in self._x2idx:
                    x_emb[i, :] = self.w_emb[self._x2idx[x]]
                elif x in self._w2v:
                    x_emb[i, :] = self._w2v[x]
                else:
                    x_emb[i, :] = self.w_emb[UNK_I]
                ret.append(x_emb)
        return ret

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
                        self._idx2y[len(self._y2idx)] = y
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

    def _get_test_w2v_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          np.array: embedding of the input word

        """
        emb_i = self._x2idx.get(a_word)
        if emb_i is None:
            if a_word in self.w2v:
                return floatX(self.w2v[a_word])
            return self.W_EMB[self.unk_w_i]
        return self.W_EMB[emb_i]

    def _load_emb(self, a_path, a_remember_word=lambda w: True,
                  a_mode=M_TEST):
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
        print("Loading embeddings from '{:s}'...".format(a_path),
              file=sys.stderr)
        word = ""
        fields = []
        first_line = True
        finput = AltFileInput(a_path, errors="replace")
        for iline in finput:
            if not iline:
                continue
            elif first_line:
                _, ndim = iline.split()
                if a_mode == M_TEST:
                    assert int(ndim) == self.ndim, \
                        "Deminsionality of word embeddings does not match" \
                        " the dimensioality this model was trained with."
                else:
                    self.ndim = int(ndim)
                    self.w_emb = np.empty((self.w_i, self.ndim))
                    self.w_emb[UNK_I, :] = 1e-2
                first_line = False
                continue
            fields = iline.split()
            word = ' '.join(fields[:-self.ndim])
            if a_remember_word(word):
                vec = floatX([float(v) for v in fields[-self.ndim:]])
                if a_mode == M_TRAIN:
                    self.w_emb[len(self._x2idx), :] = vec
                    self._x2idx[word] = len(self._x2idx)
                else:
                    self._w2v[word] = vec
        if a_mode == M_TRAIN:
            self.w_emb = floatX(self.w_emb)
            self.W_EMB = theano.shared(value=self.w_emb, name="W_EMB")

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
        if self._type == GRU:
            params, outvars = self._init_gru(invars)
        elif self._type == LSTM:
            params, outvars = self._init_lstm(invars)
        else:
            raise NotImplementedError
        self._params.extend(params)
        self.RNN = outvars[0]
        self.RNN2OUT = theano.shared(
            value=HE_UNIFORM((self.intm_dim * self._order, self.n_y)),
            name="RNN2OUT")
        self._params.append(self.RNN2OUT)
        self.OUT = TT.dot(self.RNN, self.RNN2OUT)
        self.Y_pred_probs = TT.nnet.softmax(self.OUT)
        self.Y_pred_labels = TT.argmax(self.OUT, axis=1)
        self.Y_gold_probs = TT.matrix(name="Y_gold_probs")
        self.Y_gold_labels = TT.matrix(name="Y_gold_labels")
        # use accuracy as primary cost function for choosing the model
        self._acc = TT.sum(TT.eq(self.Y_pred_labels, self.Y_gold_labels))
        # use cross-entropy as primary cost function to be optimized and
        # secondary cost function for choosing the model
        # (cf. https://en.wikipedia.org/wiki/Cross_entropy)
        # cross-entropy (own version)
        # self._loss = -TT.sum(
        #     self.Y_gold_probs * TT.log(self.Y_pred_probs)
        #     + (TT.ones_like(self.Y_gold_probs) - self.Y_gold_probs)
        #     * TT.log(TT.ones_like(self.Y_pred_probs) - self.Y_pred_probs))
        # self._loss = TT.mean(TT.nnet.categorical_crossentropy(
        #     self.Y_pred_probs, self.Y_gold_probs)
        # )
        # hinge loss
        hinge_good = self.Y_pred_probs[self.Y_gold_probs.nonzero()]
        hinge_bad = TT.max(
            self.Y_pred_probs * (TT.ones_like(self.Y_gold_probs)
                                 - self.Y_gold_probs), axis=1)
        alpha = 1e-5
        C = 3e-3
        self._loss = TT.sum(
            TT.nnet.relu(C + hinge_bad - hinge_good)) \
            + alpha * TT.sum(TT.sum(self.RNN2OUT ** 2, axis=-1), axis=-1)

    def _init_gru(self, a_invars, a_sfx="-forward"):
        """Initialize a GRU layer.

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
        intm_dim = self.intm_dim * self._order
        ones = TT.ones((intm_dim,), dtype=config.floatX)  # correct
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

        horder_dim = (self.intm_dim, intm_dim)
        if self._order == 1:
            HORDER = theano.shared(value=floatX(np.eye(horder_dim)),
                                   name="HORDER" + a_sfx)
        else:
            HORDER = theano.shared(value=floatX(ORTHOGONAL(horder_dim)),
                                   name="HORDER" + a_sfx)
            params.append(HORDER)

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
            """Recurrent GRU unit.

            Note:
              The general order of function parameters to fn is: sequences (if
              any), prior result(s) (if needed), non-sequences (if any)

            Args:
              x_ (theano.shared): input vector
              o_ (theano.shared): output vector
              h_ (list[theano.shared]): previous hidden vector(s)
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

            # z should match h_, ones should match h_, h_tilde should match h_,
            # r should match h_, wb should match h_, urz should match h_

            # x \in R^{1 x 100}
            # h_ \in R^{1 x (100 x order)}
            # W \in R^{(100 x 3) x 100}
            # U_rz \in R^{(100 x 2) x (100 x order)}
            # U_h \in R^{(100) x (100 x order)}
            # b \in R^{1 x (100 x 3)}
            # h \in R^{1 x (100 x order)}
            # w_do \in R^{(100- x 3) x 1}
            # u_rz_do \in R^{(2 x 100) x 1}
            # u_h_do \in R^{100 x 1}
            wb = TT.dot(W * w_do.dimshuffle((0, 'x')), x_.T).T + b
            # wb \in R^{1 x (3 x order x 100)}
            urz = TT.dot(U_rz * u_rz_do.dimshuffle((0, 'x')),
                         h_.T).dimshuffle(('x', 0))
            # urz \in R^{1 x (3 x 100 x order)}
            # update: z \in R^{1 x (order x 100)}
            z = TT.nnet.sigmoid(
                _slice(wb, 0, intm_dim) + _slice(urz, 0, intm_dim))
            # reset: r \in R^{1 x 100}
            r = TT.nnet.sigmoid(_slice(wb, 1, intm_dim)
                                + _slice(urz, 1, intm_dim))
            # preliminary activation: h \in R^{1 x 59}
            h_tilde = TT.tanh(_slice(wb, 2, intm_dim)
                              + TT.dot(U_h * u_h_do.dimshuffle((0, 'x')),
                                       (r * h_).T).T)
            h = TT.dot(HORDER, (z * h_ + (ones - z) * h_tilde).T)
            # return current output and memory state
            return TT.concatenate((h_[self.intm_dim:], h.flatten()),
                                  axis=0)

        m = 0
        outvars = []
        for iv, igbw in a_invars:
            m = iv.shape[0]
            ret, _ = theano.scan(_step,
                                 sequences=[iv],
                                 outputs_info=[
                                     floatX(np.zeros((
                                         self.intm_dim * self._order,)))],
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
                                           self._loss)
        if self._predict_labels is None:
            self._predict_labels = theano.function(
                [self.W_INDICES], self.Y_pred_labels,
                name="_predict_labels"
            )

        if self._predict_labels_emb is None:
            self._predict_labels_emb = theano.function(
                [self.EMB], self.Y_pred_labels,
                name="_predict_labels_emb"
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

        if self._compute_loss is None:
            self._compute_loss = theano.function(
                [self.W_INDICES, self.Y_gold_probs],
                self._loss, name="_compute_loss"
            )
        # initialize debug function
        if self._debug_nn is None:
            self._debug_nn = theano.function([self.W_INDICES],
                                             [self.Y_pred_probs],
                                             name="_debug_nn")

    def _train(self, X_train, Y_train, X_dev, Y_dev, Y_dev_stat):
        """Perform the actual training.

        Args:
          X_train (list[np.array]): training instances as embedding indices
          Y_train (list[np.array]): gold labels for the training instances
          X_dev (list[np.array]): dev set instances as embedding indices
          Y_dev (list[np.array]): gold labels for the dev set instances
          Y_dev_stat (tuple): statistics on the development set

        Returns:
          void:

        """
        train_loss = dev_f1 = dev_loss = 0.
        max_dev_f1 = dev_f1 = -1
        prev_train_loss = train_loss = min_dev_loss = dev_loss = INF
        start_time = end_time = time_delta = None
        best_params = []
        dev_predicts = None
        x_train = y_train = None
        N = len(X_train)
        xindices = np.arange(N)
        for i in xrange(MAX_ITERS):
            start_time = datetime.utcnow()
            # perform one training iteration
            train_loss = 0.
            if i % (N_RESAMPLE + 1) == 0:
                print("Resampling...", file=sys.stderr)
                rndm_samples = np.random.choice(xindices,
                                                N / 2, replace=False)
                x_train = [X_train[j] for j in rndm_samples]
                y_train = [Y_train[j] for j in rndm_samples]
            for x, (y_lbl, y_prob) in zip(x_train, y_train):
                train_loss += self._grad_shared(x, y_prob)
                self._update()
            # estimate the model on the dev set
            dev_predicts = []
            dev_f1 = dev_loss = 0.
            # temporarily deactivate dropout
            self.use_dropout.set_value(0.)
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
                # print("ce =", repr(self._compute_loss(x, y_prob)),
                #       file=sys.stderr)
                # print("self._predict_labels(x) =",
                #       repr(self._predict_labels(x)))
                dev_predicts.append(self._predict_labels(x))
                dev_loss += self._compute_loss(x, y_prob)
            dev_f1 = compute_macro_f1(self._y2idx.values(),
                                      Y_dev_stat, dev_predicts)
            # switch dropout on again
            # if i > START_DROPOUT:
            #     self.use_dropout.set_value(1.)
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).seconds
            stored = False
            if dev_f1 > max_dev_f1 or \
               (dev_f1 == max_dev_f1 and dev_loss < min_dev_loss):
                best_params = [p.get_value() for p in self._params]
                max_dev_f1 = dev_f1
                min_dev_loss = dev_loss
                stored = True
            print("Iteration {:d}:\ttrain_loss = {:f}\t"
                  "dev_f1={:f}\tdev_loss={:f}\t({:.2f} sec){:s}".format(
                      i, train_loss, dev_f1, dev_loss, time_delta,
                      " *" if stored else ''),
                  file=sys.stderr)
            if abs(prev_train_loss - train_loss) < CONV_EPS:
                break
            prev_train_loss = train_loss
        # deactivate dropout
        self.use_dropout.set_value(0.)
        if best_params:
            for p, val in zip(self._params, best_params):
                p.set_value(val)
        else:
            raise RuntimeError("Network could not be trained.")

    def _balance_ds(self, a_X, a_Y):
        """Balance data set by repeating underrepresented samples.

        Args:
          a_X (list): input instances
          a_Y (list): gold labels

        Returns:
          void:

        Note:
          modifies input lists in place

        """
        y2i = defaultdict(set)
        y2n = {}
        i2y = defaultdict(set)
        for i, y_inst in enumerate(a_Y):
            for y in y_inst:
                y2i[y].add(i)
                i2y[i].add(y)
        # determine the maximum number of instances for one label
        max_n = 0
        for y, v in y2i.iteritems():
            y2n[y] = len(v)
            max_n = max(len(v), max_n) / 2
            y2i[y] = np.array([i for i in v])
        new_X = []
        new_Y = []
        n_samples = 0
        new_samples = None
        for y, v in y2i.iteritems():
            n_samples = max(0, max_n - y2n[y])
            new_samples = np.random.choice(v, n_samples)
            for i in set(new_samples):
                for y_i in i2y[i]:
                    y2n[y_i] -= 1
                new_X.append(a_X[i])
                new_Y.append(a_Y[i])
        a_X.extend(new_X)
        a_Y.extend(new_Y)

    def _shuffle(self, a_X, a_Y):
        """Randomly shuffle data set.

        Args:
          a_X (list): input instances
          a_Y (list): gold labels

        Returns:
          2-tuple: shiffled a_X and a_Y

        """
        xlen = len(a_X)
        rndm = np.arange(xlen)
        np.random.shuffle(rndm)
        return [a_X[i] for i in rndm], [a_Y[i] for i in rndm]
