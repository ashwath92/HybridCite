#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
from numpy import zeros, float32 as REAL
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset, memcpy

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

cimport cython

# scipy <= 0.15
try:
     from scipy.linalg.blas import fblas
except ImportError:
     # in scipy > 0.15, fblas function has been removed
     import scipy.linalg.blas as fblas


#
# shared type definitions for word2vec_inner
# used by both word2vec_inner.pyx (automatically) and doc2vec_inner.pyx (by explicit cimport)
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.htmlcimport numpy as np

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cimport numpy as np
ctypedef np.float32_t REAL_t

# BLAS routine signatures

# precalculated sigmoid table
DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6
cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

ctypedef REAL_t (*our_dot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef void (*our_saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

cdef our_dot_ptr our_dot
cdef our_saxpy_ptr our_saxpy

cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)


# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]

# to support random draws from negative-sampling cum_table

DEF MAX_DOCUMENT_LEN = 10000

DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
        our_saxpy = saxpy
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random

cdef unsigned long long fast_document_dm_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, unsigned long long next_random,
    REAL_t *neu1, REAL_t *syn1neg, const int predict_word_index, const REAL_t alpha, REAL_t *work,
    const int size, int learn_hidden) nogil:

    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    # l1 already composed by caller, passed in as neu1
    # work (also passsed in) will accumulate l1 error for outside application
    for d in range(negative+1):
        if d == 0:
            target_index = predict_word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == predict_word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    return next_random

def train_hyperdocument_dm(model, doc_words, doctag_indexes, anchors, alpha, work=None, neu1=None,
                      learn_doctags=True, learn_words=True, learn_hidden=True,
                      word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
    '''
    train_document_dm(model,doc_words,doctag_indexes,alpha,work,neu1,learn_doctags,learn_words,learn_hidden,word_vectors,
                      doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
    '''
    cdef int i
    cdef int anchor_len=0
    cdef np.uint32_t anchor_pos[MAX_DOCUMENT_LEN]
    cdef np.uint32_t anchor_id[MAX_DOCUMENT_LEN]
    for i in range(len(anchors)):
        if anchors[i].doc_id not in model.docvecs.doctags:# for some bad doc in ACL dataset
            continue
        anchor_pos[anchor_len]=anchors[i].pos
        anchor_id[anchor_len]= model.docvecs.doctags[anchors[i].doc_id].offset
        anchor_len+=1

    cdef int hs = model.hs
    cdef int negative = model.anchor_negative
    cdef int sample = (model.sample != 0)
    cdef int _learn_doctags = learn_doctags
    cdef int _learn_words = learn_words
    cdef int _learn_hidden = learn_hidden
    cdef int cbow_mean = model.cbow_mean
    cdef REAL_t count, inv_count = 1.0

    cdef REAL_t *_word_vectors
    cdef REAL_t *_doctag_vectors
    cdef REAL_t *_word_locks
    cdef REAL_t *_doctag_locks
    cdef REAL_t *_work
    cdef REAL_t *_neu1
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_DOCUMENT_LEN]
    cdef np.uint32_t indexes[MAX_DOCUMENT_LEN]
    cdef np.uint32_t kept_word_pos[MAX_DOCUMENT_LEN]
    cdef np.uint32_t _doctag_indexes[MAX_DOCUMENT_LEN]
    cdef np.uint32_t reduced_windows[MAX_DOCUMENT_LEN]
    cdef int document_len
    cdef int doctag_len
    cdef int window = model.anchor_window #fix

    cdef int j, k, m
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_DOCUMENT_LEN]
    cdef np.uint8_t *codes[MAX_DOCUMENT_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    cdef unsigned long long next_random

    # default vectors, locks from syn0/doctag_syn0
    if word_vectors is None:
       word_vectors = model.wv.syn0
    _word_vectors = <REAL_t *>(np.PyArray_DATA(word_vectors))
    if doctag_vectors is None:
       doctag_vectors = model.docvecs.doctag_syn0
    _doctag_vectors = <REAL_t *>(np.PyArray_DATA(doctag_vectors))
    if word_locks is None:
       word_locks = model.syn0_lockf
    _word_locks = <REAL_t *>(np.PyArray_DATA(word_locks))
    if doctag_locks is None:
       doctag_locks = model.docvecs.doctag_syn0_lockf
    _doctag_locks = <REAL_t *>(np.PyArray_DATA(doctag_locks))


    syn1neg = <REAL_t *>(np.PyArray_DATA(model.docvecs.doctag_syn1neg))
    cum_table = <np.uint32_t *>(np.PyArray_DATA(model.docvecs.cum_table))
    cum_table_len = len(model.docvecs.cum_table)

    next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    if work is None:
       work = zeros(model.layer1_size, dtype=REAL)
    _work = <REAL_t *>np.PyArray_DATA(work)
    if neu1 is None:
       neu1 = zeros(model.layer1_size, dtype=REAL)
    _neu1 = <REAL_t *>np.PyArray_DATA(neu1)

    vlookup = model.wv.vocab
    i = 0
    for idx, token in enumerate(doc_words):
        predict_word = vlookup[token] if token in vlookup else None
        if predict_word is None:  # shrink document to leave out word
            continue  # leaving i unchanged
        if sample and predict_word.sample_int < random_int32(&next_random):
            continue
        indexes[i] = predict_word.index
        kept_word_pos[i]=idx
        if hs:
            codelens[i] = <int>len(predict_word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(predict_word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(predict_word.point)
        #result += 1
        i += 1
        if i == MAX_DOCUMENT_LEN:
            break  # TODO: log warning, tally overflow?
    document_len = i

    # single randint() call avoids a big thread-sync slowdown
    for i, item in enumerate(model.random.randint(0, window, anchor_len)):
        reduced_windows[i] = item

    doctag_len = <int>min(MAX_DOCUMENT_LEN, len(doctag_indexes))
    for i in range(doctag_len):
        _doctag_indexes[i] = doctag_indexes[i]
        #result += 1

    # release GIL & train on the document
    cdef int new_anchor_pos
    with nogil:
        for i in range(anchor_len):
            result+=1
            new_anchor_pos=bisect_left(kept_word_pos,anchor_pos[i],0,document_len)
            j = new_anchor_pos - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = new_anchor_pos + window - reduced_windows[i]
            if k > document_len:
                k = document_len

            # compose l1 (in _neu1) & clear _work
            memset(_neu1, 0, size * cython.sizeof(REAL_t))
            count = <REAL_t>0.0
            for m in range(j, k):
                count += ONEF
                our_saxpy(&size, &ONEF, &_word_vectors[indexes[m] * size], &ONE, _neu1, &ONE)

            for m in range(doctag_len):
                count += ONEF
                our_saxpy(&size, &ONEF, &_doctag_vectors[_doctag_indexes[m] * size], &ONE, _neu1, &ONE)

            if count > (<REAL_t>0.5):
                inv_count = ONEF/count
            if cbow_mean:
                sscal(&size, &inv_count, _neu1, &ONE)  # (does this need BLAS-variants like saxpy?)
            memset(_work, 0, size * cython.sizeof(REAL_t))  # work to accumulate l1 error
            if hs:
                #fast_document_dm_hs(points[i], codes[i], codelens[i], _neu1, syn1, _alpha, _work, size, _learn_hidden)
                pass
            if negative:
                next_random = fast_document_dm_neg(negative, cum_table, cum_table_len, next_random,
                                                   _neu1, syn1neg, anchor_id[i], _alpha, _work,
                                                   size, _learn_hidden)

            if not cbow_mean:
                sscal(&size, &inv_count, _work, &ONE)  # (does this need BLAS-variants like saxpy?)
            # apply accumulated error in work
            if _learn_doctags:
                for m in range(doctag_len):
                    our_saxpy(&size, &_doctag_locks[_doctag_indexes[m]], _work,
                              &ONE, &_doctag_vectors[_doctag_indexes[m] * size], &ONE)
            if _learn_words:
                for m in range(j, k):
                    our_saxpy(&size, &_word_locks[indexes[m]], _work, &ONE,
                              &_word_vectors[indexes[m] * size], &ONE)

    return result

FAST_VERSION = init()