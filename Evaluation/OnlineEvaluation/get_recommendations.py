import requests
import numpy as np
import pickle
import csv
from gensim.parsing import preprocessing
import re
from tqdm import tqdm
import contractions
from copy import deepcopy
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
hd2vmodel = HyperDoc2Vec.load('/home/ashwath/Programs/MAGCS/MAG-hyperdoc2vec/models/magcsenglish_window20.model')

from numpy.random import choice
from copy import deepcopy
from collections import Counter

# GLOBAL calculation for weights (same for all recommendations) 
# Weights: reciproacal ranks of the 500 items in each list (hd2v and bm25)
hybrid_weights = [1/(i+1) for i in range(500)]
hybrid_weights.extend(hybrid_weights)
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

def hd2v_wvindvout_recommend(context):
    """ Recommend based on the hyperdoc2vec model using IN and OUT vectors"""
    context_words_list = context.split()
    word_vocabs = [hd2vmodel.wv.vocab[w] for w in context_words_list if w in hd2vmodel.wv.vocab]
    word2_indices = [word.index for word in word_vocabs]
    # Get the sum of the IN word vectors
    l1 = np.sum(hd2vmodel.wv.syn0[word2_indices], axis=0)
    if word2_indices:
        l1 /= len(word2_indices)
    # Following hd2v code, e^(sumwvIN.docvecIN + sumwvOUT.docvecOUT)
    prob_values=exp(dot(l1, hd2vmodel.docvecs.doctag_syn1neg.T))
    prob_values = nan_to_num(prob_values)
    top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
    return [hd2vmodel.docvecs.offset2doctag[index1] for index1 in top_indices]

def solr_recommend(context):
    """ """
    num_rows = 500
    pred_docs_dict = search_solr_parse_json(context, 'mag_en_cs', 'content', num_rows)
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

def hybrid_recommend(hd2v_recommendations_wv0dv1, bm25_recommendations):
    """ Treat them as equal (50:50 importance)"""
    combined_recommendations = deepcopy(hd2v_recommendations_wv0dv1)
    weights = [1/(i+1) for i in range(500)]
    weights.extend(weights)
    weights = np.array(weights)
    combined_recommendations.extend(bm25_recommendations)
    combined_recommendations = np.array(combined_recommendations)
    draw = choice(combined_recommendations, num_picks, p=hybrid_weights)
    # Now, we have drawn 500000 times with replacement, we expect the recommended
    # ids with high probabilities to be drawn more. Recommendations in both the
    # recommendation lists (with 2 separate probabilities) will be drawn even more.
    # Create a Counter with number of times each recommended id is drawn.
    draw_frequencies = Counter(draw)
    rankedlist = [paperid for paperid, draw_count in draw_frequencies.most_common(500)]
    return rankedlist

def get_top10(recommendations_list):
    """ Returns the first 10 elements of a list"""
    return recommendations_list[:10]

def get_paper_details(recommendations_list):
    """ Gets details from the DB for the list of paper ids in the recommendations list"""
    cur.execute("""SELECT papers.paperid, originaltitle, publishedyear, citationcount, abstract 
                    FROM papers
                    LEFT JOIN  paperabstracts on papers.paperid = paperabstracts.paperid
                    WHERE papers.paperid in {}""".format(tuple(recommendations_list)))
    return cur.fetchall()

df = pd.read_csv('NLPcontexts_grouped.tsv', sep='\t', encoding='utf-8') #15446 rows.
df['ground_truth'] = df['ground_truth'].astype('str').apply(lambda x: x.split(','))
df['cleanedcontext'] = df['context'].progress_apply(clean_text)
sample_df = df.sample(n=100, random_state=8)

sample_df['hd2v_recommendations_wv0dv1'] = sample_df['cleanedcontext'].progress_apply(hd2v_wvindvout_recommend)
sample_df['bm25_recommendations'] = sample_df['cleanedcontext'].progress_apply(solr_recommend)
hybrid_models_cols = ['hd2v_recommendations_wv0dv1', 'bm25_recommendations']
sample_df['hybrid_recommendations'] = sample_df[hybrid_models_cols].progress_apply(
          lambda x: hybrid_recommend(x.hd2v_recommendations_wv0dv1, x.bm25_recommendations), axis=1)
sample_df['hybrid_recommendations'] = sample_df['hybrid_recommendations'].apply(get_top10)
sample_df['bm25_recommendations'] = sample_df['bm25_recommendations'].apply(get_top10)
sample_df['hd2v_recommendations_wv0dv1'] = sample_df['hd2v_recommendations_wv0dv1'].apply(get_top10)

#sample_df[dbcols] = sample_df['hd2v_recommendations_wv0dv1'].apply(get_paper_details, axis=1, result_type='expand')
sample_df['hd2v_recommendations_metadata'] = sample_df['hd2v_recommendations_wv0dv1'].progress_apply(get_paper_details)
sample_df['bm25_recommendations_metadata'] = sample_df['bm25_recommendations'].progress_apply(get_paper_details)
sample_df['hybrid_recommendations_metadata'] = sample_df['hybrid_recommendations'].progress_apply(get_paper_details)
sample_df.to_pickle('sample_100_top10_3models.pickle')

cols_for_json = ['context', 'hd2v_recommendations_metadata', 'bm25_recommendations_metadata', 'hybrid_recommendations_metadata']
context_recommendations = sample_df[cols_for_json].values.tolist()

jsonarray = []
for context_recommendation in context_recommendations:
    # 0 is context, 1 is hd2v_recommendations_metadata, 2 is bm25_recommendations_metadata,
    # 3 is hybrid_recommendations_metadata
    jsonarray.append({
            'context': context_recommendation[0],
            'hd2v': context_recommendation[1],
            'bm25': context_recommendation[2],
            'hybrid': context_recommendation[3]
            })

with open('online_eval_recommendations_100.json', 'w') as jsonfile:
    json.dump(jsonarray, jsonfile, indent=4)