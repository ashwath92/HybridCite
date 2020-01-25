from io import open
from gensim.models.doc2vec import *

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp

from scipy.special import expit

from numpy import nan_to_num

from bisect import bisect_left

import types

if True: from HyperDoc2vec_inner import train_hyperdocument_dm
else:
    def train_hyperdocument_dm(model, doc_words, doctag_indexes, anchors, alpha, work=None, neu1=None,
                               learn_doctags=True, learn_words=True, learn_hidden=True,
                               word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed memory model ("PV-DM") by training on a single document.

        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`. This
        method implements the DM model with a projection (input) layer that is
        either the sum or mean of the context vectors, depending on the model's
        `dm_mean` configuration field.  See `train_document_dm_concat()` for the DM
        model with a concatenated input layer.

        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.

        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.

        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        '''
        train_document_dm(model, doc_words, doctag_indexes, alpha, work, neu1, learn_doctags, learn_words, learn_hidden,
                          word_vectors,
                          doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
        '''
        if word_vectors is None:
            word_vectors = model.wv.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        model.neg_labels = []
        if model.anchor_negative > 0:
            # precompute negative labels optimization for pure-python training
            model.neg_labels = zeros(model.anchor_negative + 1)
            model.neg_labels[0] = 1.

        kept_word_idx = []
        word_vocabs = []
        for idx, w in enumerate(doc_words):
            if w in model.wv.vocab and model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32:
                kept_word_idx.append(idx)
                word_vocabs.append(model.wv.vocab[w])

        for anchor in anchors:
            if anchor.doc_id not in model.docvecs.doctags:
                continue
            reduced_window = model.random.randint(model.anchor_window)  # `b` in the original doc2vec code
            new_anchor_pos = bisect_left(kept_word_idx, anchor.pos)
            start = max(0, new_anchor_pos - model.anchor_window + reduced_window)
            windows_pos = enumerate(word_vocabs[start:(new_anchor_pos + model.anchor_window - reduced_window)], start)
            word2_indexes = [word2.index for pos2, word2 in windows_pos]
            l1 = np_sum(word_vectors[word2_indexes], axis=0) + np_sum(doctag_vectors[doctag_indexes], axis=0)
            count = len(word2_indexes) + len(doctag_indexes)
            if model.cbow_mean and count > 1:
                l1 /= count
            neu1e = train_cbow_anchor_pair(model, anchor, word2_indexes, l1, alpha,
                                           learn_vectors=False, learn_hidden=learn_hidden)
            if not model.cbow_mean and count > 1:
                neu1e /= count
            if learn_doctags:
                for i in doctag_indexes:
                    doctag_vectors[i] += neu1e * doctag_locks[i]
            if learn_words:
                for i in word2_indexes:
                    word_vectors[i] += neu1e * word_locks[i]

        return len(anchors)

def train_cbow_anchor_pair(model, anchor, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True,
                    compute_loss=False):
    neu1e = zeros(l1.shape)

    if model.hs:
        l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
        prod_term = dot(l1, l2a.T)
        fa = expit(prod_term)  # propagate hidden -> output
        ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

        # loss component corresponding to hierarchical softmax
        if compute_loss:
            sgn = (-1.0) ** word.code  # ch function, 0-> 1, 1 -> -1
            model.running_training_loss += sum(-log(expit(-sgn * prod_term)))

    if model.anchor_negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        doc_indice=model.docvecs.doctags[anchor.doc_id].offset
        doc_indices = [doc_indice]
        while len(doc_indices) < model.anchor_negative + 1:
            d = model.docvecs.cum_table.searchsorted(model.random.randint(model.docvecs.cum_table[-1]))
            if d != doc_indice:
                doc_indices.append(d)
        l2b = model.docvecs.doctag_syn1neg[doc_indices]  # 2d matrix, k+1 x layer1_size
        prod_term = dot(l1, l2b.T)
        fb = expit(prod_term)  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.docvecs.doctag_syn1neg[doc_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

        # loss component corresponding to negative sampling
        if compute_loss:
            model.running_training_loss -= sum(log(expit(-1 * prod_term[1:])))  # for the sampled words
            model.running_training_loss -= log(expit(prod_term[0]))  # for the output word

    if learn_vectors:
        # learn input -> hidden, here for all words in the window separately
        if not model.cbow_mean and input_word_indices:
            neu1e /= len(input_word_indices)
        for i in input_word_indices:
            model.wv.syn0[i] += neu1e * model.syn0_lockf[i]

    return neu1e

def _do_train_job(self, job, alpha, inits):
    work, neu1 = inits
    tally = 0
    for doc in job:
        indexed_doctags = self.docvecs.indexed_doctags(doc.tags)
        doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags
        if self.sg:
            tally += train_document_dbow(self, doc.words, doctag_indexes, alpha, work,
                                         train_words=self.dbow_words,
                                         doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
        elif self.dm_concat:
            tally += train_document_dm_concat(self, doc.words, doctag_indexes, alpha, work, neu1,
                                              doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
        else:
            if len(doc.anchors)>0:
                tally += train_hyperdocument_dm(self, doc.words, doctag_indexes, doc.anchors, alpha, work, neu1,
                                            doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
        self.docvecs.trained_item(indexed_doctags)
    return tally, self._raw_word_count(job)

class HyperDoc2Vec(Doc2Vec):
    def __init__(self, documents=None, dm_mean=None,
                 dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1,
                 docvecs=None, docvecs_mapfile=None, comment=None, trim_rule=None, docid_prefix='=-=', docid_suffix='-=-', anchor_window=5, anchor_negative=5, anchor_iter=100, anchor_alpha=0.025, **kwargs):

        self.docid_prefix=docid_prefix
        self.docid_suffix=docid_suffix
        self.anchor_window=int(anchor_window)
        self.anchor_negative=int(anchor_negative)
        self.anchor_iter=int(anchor_iter)
        self.anchor_alpha=anchor_alpha

        self.docvecs = docvecs or HyperDocvecsArray(docvecs_mapfile)
        super(HyperDoc2Vec, self).__init__(documents=documents, docvecs = self.docvecs, **kwargs)
        #self.docvecs.reset_weights(self)
        #self.reset_weights()
        self._do_train_job=types.MethodType(_do_train_job,self)

        #Jialong
        self.comment = comment
        if documents is not None:
            # jialong
            if self.negative:
                self.docvecs.make_cum_table()
            self.train(documents, total_examples=self.corpus_count, epochs=self.anchor_iter, start_alpha=self.anchor_alpha)

    def __str__(self):
        """Abbreviated name reflecting major configuration paramaters."""
        segments = []
        segments.append('i%d' % self.anchor_iter)
        segments.append('n%d' % self.anchor_negative)  # negative samples
        segments.append('w%d' % self.anchor_window)  # window size, when relevant
        segments.append('d%d' % self.vector_size)  # dimensions
        segments.append('alp%f' % self.anchor_alpha)
        return '%s(%s)' % (self.__class__.__name__, ','.join(segments))

    def predict_output_doc(self, context_words_list, source_doc_id_or_text=None, topn=10, candidates=None):
        """Report the probability distribution of the center word given the context words as input to the trained model."""
        if not self.negative:
            raise RuntimeError("We have currently only implemented predict_output_word "
                "for the negative sampling scheme, so you need to have "
                "run word2vec with negative > 0 for this to work.")

        if not hasattr(self.wv, 'syn0') or not hasattr(self.docvecs, 'doctag_syn1neg') or not hasattr(self.docvecs, 'doctag_syn0'):
            raise RuntimeError("Parameters required for predicting the output words not found.")

        word_vocabs = [self.wv.vocab[w] for w in context_words_list if w in self.wv.vocab]
        if not word_vocabs:
            warnings.warn("All the input context words are out-of-vocabulary for the current model.")

        word2_indices = [word.index for word in word_vocabs]

        validDocInput=False
        l1 = np_sum(self.wv.syn0[word2_indices], axis=0)
        if isinstance(source_doc_id_or_text,string_types):
            source_doc_id=source_doc_id_or_text
            if source_doc_id in self.docvecs.doctags:
                l1 += self.docvecs[source_doc_id]
                validDocInput=True
            else:
                warnings.warn("Doc ID not in training data.")
        elif isinstance(source_doc_id_or_text,list):
            source_doc_text=source_doc_id_or_text
            l1+=self.infer_vector(source_doc_text,alpha=0.025,steps=100)
            validDocInput=True
        if (word2_indices or validDocInput) and self.cbow_mean:
            if validDocInput:
                l1 /= len(word2_indices)+1
            else:
                l1 /= len(word2_indices)
        prob_values = exp(dot(l1, self.docvecs.doctag_syn1neg.T)) # propagate hidden -> output and take softmax to get probabilities
        prob_values = nan_to_num(prob_values)
        prob_values /= sum(prob_values)
        if candidates!=None:
            top_indices = [idx for idx in matutils.argsort(prob_values, reverse=True) if
                           self.docvecs.offset2doctag[idx] in candidates][:topn]
        else:
            top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
        return [(self.docvecs.offset2doctag[index1], prob_values[index1]) for index1 in top_indices]   #returning the most probable output words with their probabilities

    def most_similar(self, positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None):
        self.init_sims()
        self.docvecs.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative
        ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            else:
                mean.append(weight * self.wv.word_vec(word, use_norm=True))
                if word in self.wv.vocab:
                    all_words.add(self.wv.vocab[word].index)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        limited = self.docvecs.doctag_syn0norm if restrict_vocab is None else self.doctag_syn0norm[:restrict_vocab]
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.docvecs.offset2doctag[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

def double_vec_predict_output_doc(self, context_words_list, source_doc_id_or_text=None, topn=10, candidates=None):
    """Report the probability distribution of the center word given the context words as input to the trained model."""
    if not self.negative:
        raise RuntimeError("We have currently only implemented predict_output_word "
            "for the negative sampling scheme, so you need to have "
            "run word2vec with negative > 0 for this to work.")

    if not hasattr(self.wv, 'syn0') or not hasattr(self.docvecs, 'doctag_syn1neg') or not hasattr(self.docvecs, 'doctag_syn0'):
        raise RuntimeError("Parameters required for predicting the output words not found.")

    word_vocabs = [self.wv.vocab[w] for w in context_words_list if w in self.wv.vocab]
    if not word_vocabsx:
        warnings.warn("All the input context words are out-of-vocabulary for the current model.")

    word2_indices = [word.index for word in word_vocabs]

    validDocInput=False
    l1 = np_sum(self.wv.syn0[word2_indices], axis=0)
    l2 = np_sum(self.syn1neg[word2_indices], axis=0)
    if word2_indices:
        l2/=len(word2_indices)
    if isinstance(source_doc_id_or_text,string_types):
        source_doc_id=source_doc_id_or_text
        if source_doc_id in self.docvecs.doctags:
            l1 += self.docvecs[source_doc_id]
            validDocInput=True
        else:
            warnings.warn("Doc ID not in training data.")
    elif isinstance(source_doc_id_or_text,list):
        source_doc_text=source_doc_id_or_text
        l1+=self.infer_vector(source_doc_text,alpha=0.025,steps=100)
        validDocInput=True
    if (word2_indices or validDocInput) and self.cbow_mean:
        if validDocInput:
            l1 /= len(word2_indices)+1
        else:
            l1 /= len(word2_indices)
    #prob_values = exp(dot(l1, self.docvecs.doctag_syn1neg.T)) # propagate hidden -> output and take softmax to get probabilities
    prob_values=exp(dot(l1, self.docvecs.doctag_syn1neg.T)+dot(l2,self.docvecs.doctag_syn0.T))
    prob_values = nan_to_num(prob_values)
    prob_values /= sum(prob_values)
    if candidates!=None:
        top_indices = [idx for idx in matutils.argsort(prob_values, reverse=True) if
                       self.docvecs.offset2doctag[idx] in candidates][:topn]
    else:
        top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
    return [(self.docvecs.offset2doctag[index1], prob_values[index1]) for index1 in top_indices]   #returning the most probable output words with their probabilities

class HyperDocvecsArray(DocvecsArray):
    def __init__(self, mapfile_path=None):
        super(HyperDocvecsArray,self).__init__(mapfile_path)
        self.cum_table=None

    def __len__(self):
        return max(1,self.count)

    def init_sims(self, replace=False):
        super(HyperDocvecsArray, self).init_sims(replace)
        if getattr(self, 'doctag_syn1norm', None) is None or replace:
            logger.info("precomputing L2-norms of doc weight vectors (OUT)")
            if replace:
                for i in xrange(self.doctag_syn1neg.shape[0]):
                    self.doctag_syn1neg[i, :] /= sqrt((self.doctag_syn1neg[i, :] ** 2).sum(-1))
                self.doctag_syn1norm = self.doctag_syn1neg
            else:
                if self.mapfile_path:
                    self.doctag_syn1norm = np_memmap(
                        self.mapfile_path+'.doctag_syn1norm', dtype=REAL,
                        mode='w+', shape=self.doctag_syn1.shape)
                else:
                    self.doctag_syn1norm = empty(self.doctag_syn1neg.shape, dtype=REAL)
                np_divide(self.doctag_syn1neg, sqrt((self.doctag_syn1neg ** 2).sum(-1))[..., newaxis], self.doctag_syn1norm)
                self.doctag_syn1norm=nan_to_num(self.doctag_syn1norm)

    def clear_sims(self):
        super(HyperDocvecsArray, self).clear_sims()
        self.doctag_syn1norm=None

    def reset_weights(self, model):
        super(HyperDocvecsArray, self).reset_weights(model)
        length = max(len(self.doctags), self.count)

        if model.hs:
            self.doctag_syn1 = zeros((length, model.layer1_size), dtype=REAL)
        if model.negative:
            self.doctag_syn1neg = zeros((length, model.layer1_size), dtype=REAL)
        self.doctag_syn1_lockf = ones((length,), dtype=REAL)  # zeros suppress learning
        
    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        corpus_size = len(self.doctags)
        self.cum_table = zeros(corpus_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for doc_index in xrange(corpus_size):
            train_words_pow += self.doctags[self.offset2doctag[doc_index]].doc_count**power
        cumulative = 0.0
        for doc_index in xrange(corpus_size):
            cumulative += self.doctags[self.offset2doctag[doc_index]].doc_count**power
            self.cum_table[doc_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

class Anchor(namedtuple('Anchor', 'doc_id pos')):
    def __str__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.doc_id, self.pos)

class TaggedHyperDocument:
    """
    A single document, made up of `words` (a list of unicode string tokens)
    and `tags` (a list of tokens). Tags may be one or more unicode string
    tokens, but typical practice (which will also be most memory-efficient) is
    for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from Word2Vec.

    """
    def __str__(self):
        return '%s(%s, %s, %s)' % (self.__class__.__name__, self.words, self.tags, self.anchors)

    def __init__(self, body, tags, docid_prefix = '=-=', docid_suffix = '-=-'):
        self.tags=tags
        self.words=[]
        self.anchors=[]
        docid_cnt=0
        for pos, word in enumerate(body):
            if word.startswith(docid_prefix) and word.endswith(docid_suffix):
                anchors=word[len(docid_prefix):-len(docid_suffix)]
                for anchor in anchors.split(','):
                    self.anchors.append(Anchor(anchor, pos-docid_cnt))
            else:
                self.words.append(word)

class TaggedLineHyperDocument(object):
    """Simple format: one document = one line = one TaggedDocument object.

    Words are expected to be already preprocessed and separated by whitespace,
    tags are constructed automatically from the document line number."""
    def __init__(self, source, cache=True):
        """
        `source` can be either a string (filename) or a file object.

        Example::

            documents = TaggedLineDocument('myfile.txt')

        Or for compressed files::

            documents = TaggedLineDocument('compressed_text.txt.bz2')
            documents = TaggedLineDocument('compressed_text.txt.gz')

        """
        self.cache = cache
        if cache:
            self.data=[]
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    parts = utils.to_unicode(line).split()
                    self.data.append(TaggedHyperDocument(parts[1:], [parts[0]]))
        else:
            self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""
        if self.cache:
            for doc in self.data:
                yield doc
        else:
            try:
                # Assume it is a file-like object and try treating it as such
                # Things that don't have seek will trigger an exception
                self.source.seek(0)
                for item_no, line in enumerate(self.source):
                    parts=utils.to_unicode(line).split()
                    yield TaggedHyperDocument(parts[1:], [parts[0]])
            except AttributeError:
                # If it didn't work like a file, use it as a string filename
                with utils.smart_open(self.source) as fin:
                    for item_no, line in enumerate(fin):
                        parts = utils.to_unicode(line).split()
                        yield TaggedHyperDocument(parts[1:], [parts[0]])

