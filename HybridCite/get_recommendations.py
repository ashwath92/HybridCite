import requests
import numpy as np
import pickle
import csv
from gensim.parsing import preprocessing
import re
from tqdm import tqdm
import contractions
tqdm.pandas()
import os
import pandas as pd
import json
import psycopg2
import psycopg2.extras
conn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=shetland.informatik.uni-freiburg.de")
cur = conn.cursor()
query="""SELECT originaltitle, publishedyear, citationcount FROM papers WHERE paperid in {}"""

query = """SELECT originaltitle, publishedyear, citationcount, abstract
           FROM papers LEFT JOIN paperabstracts on papers.paperid = paperabstracts.paperid
           WHERE papers.paperid=%(paperid)s
        """
from HyperDoc2Vec import *
from numpy.random import choice
from copy import deepcopy
from collections import Counter

# Order is hd2vOUT, bm25, bm25cited (weights 12.5% (1/8), 12.5% (1/8), 75% (3/4)
hybrid_weights = [(1/((i+1)*5)) for i in range(500)]
hybrid_weights.extend(hybrid_weights)
bm25cited_weights = [(3/((i+1)*5)) for i in range(500)]
hybrid_weights.extend(bm25cited_weights)
hybrid_weights = np.array(hybrid_weights)
# Convert to probabilities
hybrid_weights = hybrid_weights/hybrid_weights.sum()
# GLOBAL num_items_to_pick (with replacement) -- high number: half a million
num_picks = 1000000

topn = 500
def clean_text(text):
    """ Cleans the text in the only argument in various steps 
    ARGUMENTS: text: content/title, string
    RETURNS: cleaned text, string"""
    # Replace newlines by space. We want only one doc vector.
    text = text.replace('\n', ' ').lower()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Remove stop words
    text = preprocessing.remove_stopwords(text)
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_non_alphanum(text))
    return text

def hd2v_wvindvout_recommend(context, hd2vmodel):
    """ Recommend based on the hyperdoc2vec model using IN and OUT vectors"""
    context_words_list = context.split()
    word_vocabs = [hd2vmodel.wv.vocab[w] for w in context_words_list if w in hd2vmodel.wv.vocab]
    if word_vocabs  == []:
       return None
    word2_indices = [word.index for word in word_vocabs]
    # Get the sum of the IN word vectors
    # increase float size to avoid overflow in exp.
    l1 = np.sum(hd2vmodel.wv.syn0[word2_indices], axis=0, dtype=np.float128)
    if word2_indices:
        l1 /= len(word2_indices)
    # Following hd2v code, e^(sumwvIN.docvecIN + sumwvOUT.docvecOUT)
    prob_values=exp(dot(l1, hd2vmodel.docvecs.doctag_syn1neg.T))
    prob_values = nan_to_num(prob_values)
    top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
    return [hd2vmodel.docvecs.offset2doctag[index1] for index1 in top_indices]

def solr_recommend(context, collection):
    """ Gets 500 recommendations for the context (param) from the solr collection (param)"""
    num_rows = 500
    pred_docs_dict = search_solr_parse_json(context, collection, 'content', num_rows)
    # Get only ids
    if pred_docs_dict is None:
        return None 
    pred_docs_list = [doc['paperid'] for doc in pred_docs_dict]
    if pred_docs_list == []:
        return None
    return pred_docs_list

def solr_cited_recommend(context, collection):
    """ Gets 500 recommendations for the context (param) from the solr collection (param)"""
    num_rows = 500
    #sleep(0.2)
    pred_docs_dict = search_solr_parse_json(context, 'mag_en_cs_cited', 'content', num_rows)
    #sleep(1)
    # Get only ids
    if pred_docs_dict is None:
        return None 
    pred_docs_list = [doc['paperid'] for doc in pred_docs_dict]
    return pred_docs_list

def search_solr_parse_json(query, collection, search_field, num_rows):
    """ Searches the nounphrases collection on query,
    parses the json result and returns it as a list of dictionaries where
    each dictionary corresponds to a record. 
    ARGUMENTS: query, string: the user's query entered in a search box
               (if it is comma-separated, only one part of the query is sent
               to this function).
               collection: the Solr collection name (=nounphrases)
               search_field: the Solr field which is queried (=phrase)
    RETURNS: docs, list of dicts: the documents (records) returned by Solr 
             AFTER getting the JSON response and parsing it."""
    solr_url = 'http://localhost:8983/solr/' + collection + '/select'
    # Exact search only
    #query = '"' + query + '"'
    url_params = {'q': query, 'rows': num_rows, 'df': search_field}
    solr_response = requests.get(solr_url, params=url_params)
    #print(solr_response.url)
    if solr_response.ok:
        data = solr_response.json()
        docs = data['response']['docs']
        return docs
    else:
        print("Invalid response returned from Solr for", solr_response)
        return None
        #sys.exit(11)

def hybrid_recommend(hd2v_recommendations_wv0dv1, bm25_recommendations, bm25cited_recommendations):
    """ Hybrid recommender v2. Treat them as 20%, 20%, 60%"""
    combined_recommendations = deepcopy(hd2v_recommendations_wv0dv1)
    combined_recommendations.extend(bm25_recommendations)
    combined_recommendations.extend(bm25cited_recommendations)
    combined_recommendations = np.array(combined_recommendations)
    draw = choice(combined_recommendations, num_picks, p=hybrid_weights)
    # Now, we have drawn 1 million times with replacement, we expect the recommended
    # ids with high probabilities to be drawn more. Recommendations in both the
    # recommendation lists (with 2 separate probabilities) will be drawn even more.
    # Create a Counter with number of times each recommended id is drawn.
    draw_frequencies = Counter(draw)
    rankedlist = [paperid for paperid, draw_count in draw_frequencies.most_common(500)]
    return rankedlist

def get_topn(recommendations_list, n):
    """ Returns the first 10 elements of a list"""
    return recommendations_list[:n]

def get_paper_details(recommendations_list):
    """ Gets details from the DB for the list of paper ids in the recommendations list"""
    cur.execute("""SELECT papers.paperid, originaltitle, publishedyear, citationcount, abstract 
                    FROM papers
                    LEFT JOIN  paperabstracts on papers.paperid = paperabstracts.paperid
                    WHERE papers.paperid in {}""".format(tuple(recommendations_list)))
    return cur.fetchall()

