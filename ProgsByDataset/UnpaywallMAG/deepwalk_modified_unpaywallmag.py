## Created on 16th May 2016
## Soumyajit Ganguly
## soumyajit.ganguly@research.iiit.ac.in
## IIIT - Hyderabad, India

# Ashwath: changed to use mag

import getWalks as gw
import multiprocessing
import numpy as np
import logging
import gensim
import time
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
CORES = multiprocessing.cpu_count()

FLAG_preTrained = True

def getContext(graph_file, num_walks, walk_size):
    # build a graph from the references file. I think it's in the form: Citing Paper, cited paper
    # load_edgelist connects edges to form a graph, edges are undirected, so edges in both directions
    # are added from the file.
    graph = gw.load_edgelist(graph_file) 
    # At the end of load_edgelist: G[ID1] = id2, G[id2] = id1
    # This happens for every row in the 2k_sim file.
    # walk_size = 40, make 40 random jumps from the source node.
    random_walks = gw.build_deepwalk_corpus(graph, num_walks, walk_size)
    # random_walks is a list of lists: #nodes lists of lists, where each inner list contains the nodes
    # reached by a random walk from the first node (paper).
    return random_walks
    #writeAsText(random_walks)

def train(model_params):
    sentences = getContext(model_params['graph_file'], model_params['num_walks'], model_params['walk_size'])
    # sentences contains a list of list of integer paperids (where the first element of each inner list is a node from which
    # all the other nodes in the list can be reached by random walks)
    # So, we have a list of papers in which neighbours co-occur with each other
    # cbow by default: predict centre from context
    print("training word2vec")
    model = gensim.models.Word2Vec(size=model_params['vec_size'], window=model_params['window'], alpha=0.0025,
                                   min_count=0, workers=CORES)
    # Normally, this is a list of strings. But here we have a list of paper ids
    print("building vocab")
    model.build_vocab(sentences)
    if FLAG_preTrained:
        print("Intersecting word2vec format")
        # SO WHAT WE ARE DOING HERE IS INTERSECTING THE 'MEANING' OF EACH DOCUMENT WITH THE WORD2VEC VECTORS FORMED FROM THE
        #  NODES WHICH CAN BE REACHED FROM THE SAME DOCUMENT IN A RANDOM WALK
        # intersect word2vec format will bering in only those vectors from embd_file (i.e., doc2vec) which are aleready present in
        # the current word vocabulary (paper id vocab in other words)
        # So this create links between similar papers which are not already linked by citations, in the authors' words.
        # Merge in an input-hidden weight matrix loaded from the original C word2vec-tool format, where it intersects with the current vocabulary.
        # No words are added to the existing vocabulary, but intersecting words adopt the file’s weights, and non-intersecting words are left alone.
        # so the intersecting papers just take the weights from doc2vec
        model.intersect_word2vec_format(model_params['embd_file'])
        # Make the model updateable by setting syn0_lockf to 1
        model.syn0_lockf = np.ones(len(model.syn0_lockf), dtype='float32')
        # https://groups.google.com/forum/#!msg/gensim/vkWH_1WD7ks/p5Pk3r-iKAAJ
        # https://github.com/RaRe-Technologies/gensim/blob/4fb424c8649cc46be780dfc051a3e3f31e1a978f/gensim/models/word2vec.py#L1120
        # Interesting thread: https://gensim.narkive.com/XBC61PvM/gensim-3622-doc2vec-inference-stage

    # Predict the centre paper id from the vectors of the surrounding papers (not their context, but their doc embeddings)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('paper2vecmodel_unpaywallmag.dat')

    return model

def main(model_params):
    #for i in range(1, 16):
    #for k in range(1, 10):
    model = train(model_params)
    model.wv.save_word2vec_format('Paper2Vec_unpaywallmag.txt', binary=False)
    #print(model.wv.most_similar(paper_id))


if __name__ == '__main__':
    # paperrefs_cs.tsv is taken from the PaperReferences file and only includes computer science papers. The filtering is done
    # in pandaspostgres.py. Note that this also discards references (cited papers) which are not citing papers.
    model_params = {'graph_file': '/home/ashwath/Programs/UnpaywallMAG/AdditionalOutputs/unpaywallmag_references_may7.tsv',
                    'num_walks': 10, 'walk_size': 40, 'embd_file': '/home/ashwath/Programs/UnpaywallMAG/Paper2Vec/UnpaywallMagd2v.txt',
                    'vec_size': 500, 'window': 5}
    main(model_params)
    #getContext(model_params['graph_file'], model_params['num_walks'], model_params['walk_size'])
