# https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf has the formulae
import numpy as np
import pickle
#try:
#    import modin.pandas as pd
#except ModuleNotFoundError:
#    import pandas as pd
import pandas as pd
import sys, os
#sys.path.append(os.path.abspath('/home/ashwath/Programs/ArxivCS/hyperdoc2vec_arxivmag'))
import psycopg2
import psycopg2.extras
from time import time, sleep
import requests
import re
import subprocess
import spacy
from tqdm import tqdm
tqdm.pandas()
import contractions
from nltk.stem import SnowballStemmer
from gensim.parsing import preprocessing
from gensim import matutils
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet
# If mallet doesn't work, use normal LDA.
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities
#import dask.dataframe as dd
from HyperDoc2Vec import *

from numpy.random import choice
from copy import deepcopy
from collections import Counter

# GLOBAL calculation for weights (same for all recommendations) 

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

snowball = SnowballStemmer(language='english')
nlp = spacy.load('en', disable=['parser', 'ner'])
nlp.Defaults.stop_words |= {'table', 'ref', 'formula', 'citation', 'cit', 'references'
                            'fig', 'figure', 'abstract', 'introduction',
                            'description','conclusion','results','discussion'}
mallet_path = '/home/ashwath/mallet-2.0.8/bin/mallet'

# LOAD MODELS
loadmodstart = time()
id2word_dictionary = corpora.Dictionary.load('/home/ashwath/Programs/MAGCS/LDA/mag.dict')
corpus = corpora.MmCorpus('/home/ashwath/Programs/MAGCS/LDA/mag_bow_corpus.mm')
try:
    ldamallet = LdaMallet.load('/home/ashwath/Programs/MAGCS/LDA/ldamallet_mag.model')
    vec_bow_test = id2word_dictionary.doc2bow(['test'])
    vec_ldamallet = ldamallet[vec_bow_test]
except subprocess.CalledProcessError:
    print("LDA MALLET COULDN'T READ INSTANCE FILE. USING NORMAL LDA INSTEAD")
    ldamallet = LdaModel.load('/home/ashwath/Programs/MAGCS/LDA/lda_mag.model')
    
#malletindex = similarities.MatrixSimilarity.load('/home/ashwath/Programs/MAGCS/LDA/simIndexMag.index')
with open('/home/ashwath/Programs/MAGCS/LDA/docid_to_magid_training_mag.pickle', 'rb') as pick:
    docid_to_magid = pickle.load(pick)

hd2vmodel = HyperDoc2Vec.load('/home/ashwath/Programs/MAGCS/MAG-hyperdoc2vec/models/magcsenglish_window20.model')
print("MODELS took {} seconds to load".format(time()-loadmodstart))

def remove_stopwords(context):
    #print("Removing stop words.")
    return [word for word in simple_preprocess(context) if word not in nlp.Defaults.stop_words] 

def snowballstem(context_word_list):
    """Use nltk and Snowball stemmer to stem."""
    #print("Stemming using Snowball Stemmer")
    #texts_gen = back_to_string(texts)
    # KEEP ONLY NOUNS, ADJ, VERB, ADV
    return [snowball.stem(word) for word in context_word_list]

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

def binarize_predictions(relevant_list, predicted_list, importance=True):
    """ importance=True if the first ground truth is more important than the others"""
    #print(predicted_list)
    """Returns 2 if the first entry is present in the predictions, 1 if one of the
    other relevant items is present in the predictions, 0 otherwise."""
    #bin_list = []
    #for ground_truth, pred in zip(relevant_list, predicted_list):
    if importance:
        return [2 if entry == relevant_list[0] else 1 if entry in relevant_list[1:] else 0 
                for entry in predicted_list]
    else:
        return [1 if entry in relevant_list else 0 for entry in predicted_list]

def precision_at_k(predicted_bin_list, k):
    """Gets the precision at k: true positives/true positives + false positives
    Not a ranking metric, order doesn't matter."""
    # % relevant in predicted_list[:k]. bin_list has 1s and 0s
    predicted_bin_list_k = predicted_bin_list[:k]
    #predicted_true_or_false = [1 if item in relevant_list else 0 for item in predicted_bin_list_k]
    return np.sum(predicted_bin_list_k) / k

def average_precision(predicted_bin_list, k):
    """ Avg. precision = avg. of P@K for each relevant doc found in predicted_list (rank of that doc)
    1 0 1 0 1: 1/3(1/1 + 2/3 + 3/5). If only 1 relevant doc, AP = P@K. If 2 relevant docs, 1 is present at
    4th pos, 1/2(1/4+0)
    """
    # We don't want 2s for precision and ap
    predicted_bin_list_k = predicted_bin_list[:k]
    predicted_bin_list_k = [1 if entry>0 else 0 for entry in predicted_bin_list_k]
    precisions = [precision_at_k(predicted_bin_list_k, i+1) for i, item in enumerate(predicted_bin_list_k) if item > 0]
    if precisions == []:
        return 0
    #print(precisions)
    return np.sum(precisions)/np.sum(predicted_bin_list_k)

def recall_at_k(predicted_bin_list, relevant_list, k):
    """ Gets the recall at k: true positives/true positives + false negatives"""
    # how many of the relevant docs are actually present in predicted bin list
    predicted_bin_list = [1 if entry>0 else 0 for entry in predicted_bin_list]
    predicted_bin_list_k = predicted_bin_list[:k]
    #print(predicted_bin_list, relevant_list)
    num_relevant_items = len(relevant_list)
    try:
        return np.sum(predicted_bin_list_k)/num_relevant_items
    except ZeroDivisionError:
        return 0

def reciprocal_rank(predicted_bin_list, k):
    """ Reciprocal rank = 1/rank of first 'hit', i.e. first 1 in predicted_bin_list[:k]. If there is no hit, 
    it is 0."""
    predicted_bin_list_k = predicted_bin_list[:k]
    # Keep only 1s and 0s, discard 2s (2s are useful only for dcg).
    predicted_bin_list_k = [1 if entry>0 else 0 for entry in predicted_bin_list_k]
    # Get the index of the first 1
    try:
        # +1 as index starts with 0.
        rr = 1 / (predicted_bin_list_k.index(1) + 1)
        return rr
    except ValueError:
        return 0

def discounted_cumulative_gain(predicted_bin_list, k):
    """ Calculates the discounted cumulative gain for the binary list with 0s, 1s and 2s (2 is the most important
    citation, the first citation in the input file)"""
    # Get the discounted gains first: score/log rank(base 2 so that positions 1 and 2 are equally imp)
    # Convert to numpy array
    predicted_bin_array = np.array(predicted_bin_list)
    predicted_bin_array = predicted_bin_array[:k]
    # TAKE CARE: np.log2(1) = 0, we want it to be 1.
    # Element-wise division, first term is not divided by log 1, but by 1 instead.
    # it is a cumulative sum
    return predicted_bin_array[0] + np.sum(predicted_bin_array[1:] / np.log2(np.arange(2, k+1)))

def ndcg(predicted_bin_list, k):
    """ Get the normalized DCG, with the DCG values normalized by the 'ideal DCG' obtained by putting
    the most important elements at the top of the list. It is V.V. Important to note that the ideal
    DCG is obtained by sorting the top 500 elements and not all the elements."""
    # Get the ideal dcg: with the most important term (if present) at the top of the list (rank 1),
    # the other important ones after that, the 0s at the end.
    dcg_ideal = discounted_cumulative_gain(sorted(predicted_bin_list, reverse=True), k)
    if dcg_ideal == 0:
    #if not dcg_ideal:
        return 0
    # scalar/scalar below
    return discounted_cumulative_gain(predicted_bin_list, k) / dcg_ideal

def lda_preprocessing(context):
    """ Performs a set of preprocessing steps for LDA."""
    #bigram = Phrases([context]) # higher threshold fewer phrases.
    #bigram_mod = Phraser(bigram)
    data_nostops = remove_stopwords(context.lower())
    # Form Bigrams
    #data_bigrams = make_bigrams(data_nostops)
    #data_lemmatized = lemmatization(data_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN'])
    # Use a Snowball stemmer, lemmatization takes too much time and CPU
    data_stemmed = snowballstem(data_nostops)
    # Return a list
    return data_stemmed
    #return ' '.join(data_stemmed)

def lda_recommend(context_list):
    """ With multiprocessing using Dask"""
    #print("Recommending")
    topn = 500
    #sleep(0.2)
    vec_bow = id2word_dictionary.doc2bow(context_list)
    # This line takes a LONG time: it has to map to each of the 300 topics
    vec_ldamallet = ldamallet[vec_bow]
    # Convert the query to LDA space
    sims = malletindex[vec_ldamallet]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])[:topn]
    # sims is a list of tuples of (docid -- line num in original training file, probability)
    return [docid_to_magid.get(docid) for docid, prob in sims] 
topn = 500
    
def hd2v_recommend(context):
    """ Recommend based on the hyperdoc2vec model using IN and OUT vectors"""
    sleep(0.08)
    context_words_list = context.split()
    word_vocabs = [hd2vmodel.wv.vocab[w] for w in context_words_list if w in hd2vmodel.wv.vocab]
    word2_indices = [word.index for word in word_vocabs]
    # Get the sum of the IN word vectors
    l1 = np.sum(hd2vmodel.wv.syn0[word2_indices], axis=0)
    # And the sum of the OUT word vectors
    l2 = np.sum(hd2vmodel.syn1neg[word2_indices], axis=0)
    if word2_indices:
        l2/=len(word2_indices)
        l1 /= len(word2_indices)
    # Following hd2v code, e^(sumwvIN.docvecIN + sumwvOUT.docvecOUT)
    prob_values=exp(dot(l1, hd2vmodel.docvecs.doctag_syn1neg.T)+dot(l2,hd2vmodel.docvecs.doctag_syn0.T))
    prob_values = nan_to_num(prob_values)
    top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
    return [hd2vmodel.docvecs.offset2doctag[index1] for index1 in top_indices]

def hd2v_wvindvout_recommend(context):
    """ Recommend based on the hyperdoc2vec model using IN and OUT vectors"""
    sleep(0.08)
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
    sleep(0.08)
    pred_docs_dict = search_solr_parse_json(context, 'mag_en_cs', 'content', num_rows)
    #sleep(1)
    # Get only ids
    if pred_docs_dict is None:
        return None 
    pred_docs_list = [doc['paperid'] for doc in pred_docs_dict]
    return pred_docs_list

def solr_cited_recommend(context):
    """ """
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

def hybrid_recommend(hd2v_recommendations_wv0dv1, bm25_recommendations):
    """ Treat them as equal (50:50 importance)"""
    combined_recommendations = deepcopy(hd2v_recommendations_wv0dv1)
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

def hybrid_recommend(hd2v_recommendations_wv0dv1, bm25_recommendations, bm25cited_recommendations):
    """ Hybrid recommender v2. Treat them as 10%, 10%, 80%"""
    combined_recommendations = deepcopy(hd2v_recommendations_wv0dv1)
    combined_recommendations.extend(bm25_recommendations)
    combined_recommendations.extend(bm25cited_recommendations)
    combined_recommendations = np.array(combined_recommendations)
    draw = choice(combined_recommendations, num_picks, p=hybrid_weights)
    # Now, we have drawn 500000 times with replacement, we expect the recommended
    # ids with high probabilities to be drawn more. Recommendations in both the
    # recommendation lists (with 2 separate probabilities) will be drawn even more.
    # Create a Counter with number of times each recommended id is drawn.
    draw_frequencies = Counter(draw)
    rankedlist = [paperid for paperid, draw_count in draw_frequencies.most_common(500)]
    return rankedlist

def predict(filename):
    """ Get the recommendations using the 3 methods and put them in a dataframe"""
    df = pd.read_csv(filename, sep='\t', names=['ground_truth', 'citing_mag_id', 'context'], encoding='utf-8')
    print("Read file")
    #df = df.head()
    #df = df[~df.duplicated()]
    df = df[df.ground_truth.isin(hd2vmodel.docvecs.offset2doctag)]
    # Convert cited mag ids to a list
    df['ground_truth'] = df['ground_truth'].astype('str').apply(lambda x: x.split(','))
    df['context_for_lda'] = df['context'].progress_apply(lda_preprocessing)
    print("Created lda contexts")
    #sleep(0.3)
    # clean_text is present in hyperdoc2vec
    df['context'] = df['context'].progress_apply(clean_text)
    print('Cleaned contexts')
    #sleep(0.3)
    df['wordcount'] = df['context'].progress_apply(lambda x: len(x.split()))
    # Remove contexts with less than 8 words
    df = df[df.wordcount>8] # 168700 after removing
    df['hd2v_recommendations'] = df['context'].progress_apply(hd2v_recommend)
    df['hd2v_recommendations_wv0dv1'] = df['context'].progress_apply(hd2v_wvindvout_recommend)
    print('hd2v recommendations done')
    sleep(0.3)
    df['bm25_recommendations'] = df['context'].progress_apply(solr_recommend)
    #sleep(0.3)
    print('solr recommendations done')
    # LDA Recommendations take a long time, parallelize
    df['lda_recommendations'] = df['context_for_lda'].progress_apply(lda_recommend)
    print('lda recommendations done')
    df['citedbm25_recommendations'] = df['context'].progress_apply(solr_cited_recommend)
    df['citedbm25_binary'] = df[['ground_truth', 'citedbm25_recommendations']].apply(
        lambda x: binarize_predictions(x.ground_truth, x.citedbm25_recommendations, importance=False), axis=1)
    hybrid_models_cols = ['hd2v_recommendations_wv0dv1', 'bm25_recommendations']
    df['hybrid_recommendations'] = df[hybrid_models_cols].progress_apply(
          lambda x: hybrid_recommend(x.hd2v_recommendations_wv0dv1, x.bm25_recommendations), axis=1)
    df['hybrid_recommendations_v2'] = df[hybrid_models_cols_v2].progress_apply(
        lambda x: hybrid_recommend_v2(x.hd2v_recommendations_wv0dv1, x.bm25_recommendations, x.citedbm25_recommendations), axis=1)
    df['hd2v_wv0dv1_binary'] = df[['ground_truth', 'hd2v_recommendations_wv0dv1']].apply(
        lambda x: binarize_predictions(x.ground_truth, x.hd2v_recommendations_wv0dv1, importance=False), axis=1)
    #sleep(0.3)
    df['hd2v_binary'] = df[['ground_truth', 'hd2v_recommendations']].apply(
        lambda x: binarize_predictions(x.ground_truth, x.hd2v_recommendations, importance=False), axis=1)
    #print(df.hd2v_binary)
    sleep(0.3)
    #df['lda_binary'] = df[['ground_truth', 'lda_recommendations']].apply(
    #    lambda x: binarize_predictions(x.ground_truth, x.lda_recommendations, importance=True), axis=1)
    sleep(0.3)
    df['bm25_binary'] = df[['ground_truth', 'bm25_recommendations']].apply(
        lambda x: binarize_predictions(x.ground_truth, x.bm25_recommendations, importance=False), axis=1)
    df['hybrid_binary'] = df[['ground_truth', 'hybrid_recommendations']].apply(
        lambda x: binarize_predictions(x.ground_truth, x.hybrid_recommendations, importance=False), axis=1)
    df['hybrid_binary_v2'] = df[['ground_truth', 'hybrid_recommendations_v2']].apply(
        lambda x: binarize_predictions(x.ground_truth, x.hybrid_recommendations_v2, importance=False), axis=1)
    # NOTE: for 500 recommendations, ground truth is present in hd2v's recommendations 909 times, lda 1146 times, bm25 1543 times.
    # (total: 2819) -- 32.24%, 40.65%, 54.74%
    #  t = df.bm25_binary.apply(lambda x: 1 in x or 2 in x)
    print("Binarized")
    df.to_pickle('/home/ashwath/Programs/MAGCS/Pickles/recommendations_mag_bothhd2v_bm25_500_hybrid_bm25cited_hybridv2_june23_df.pickle')
    df.to_csv('/home/ashwath/Programs/MAGCS/Pickles/recommendations_mag_bothhd2v_bm25_hybrid_500_notpickle.tsv', sep='\t')
    print("Prediction done")
    calculate_metrics(df[['hd2v_recommendations', 'bm25_recommendations', 'hd2v_recommendations_wv0dv1',
                          'hd2v_binary', 'hd2v_wv0dv1_binary', 'bm25_binary', 'hybrid_recommendations', 'hybrid_binary', 'ground_truth']])
    calculate_citedbm25_metrics(df[['citedbm25_recommendations', 'citedbm25_binary', 'ground_truth']])
    calculate_hybrid_metrics_v2(df[['hybrid_recommendations_v2', 'hybrid_binary_v2', 'ground_truth']])
    
def calculate_hybrid_metrics_v2(df):
    """ Calculates metrics at different k (1 to 10 + 20,30,40,50)"""
    #print(df.columns)
    klist = list(range(1, 11))
    klist.extend([20, 30, 40, 50, 100, 200, 300, 500])
    print(klist)
    # 14 x 3 x 4 columns added for each
    for k in tqdm(klist):
        df['average_precision_hybrid_v2_{}'.format(k)] = df['hybrid_binary_v2'].progress_apply(lambda x: average_precision(x, k))
        df['recall_hybrid_v2_{}'.format(k)] = df[['hybrid_binary_v2', 'ground_truth']].apply(
            lambda x: recall_at_k(x.hybrid_binary_v2, x.ground_truth, k), axis=1)
        df['reciprocal_rank_hybrid_v2_{}'.format(k)] = df['hybrid_binary_v2'].progress_apply(lambda x: reciprocal_rank(x, k))
        df['ndcg_hybrid_v2_{}'.format(k)] = df['hybrid_binary_v2'].progress_apply(lambda x: ndcg(x, k))
    #df.to_pickle('/home/ashwath/Programs/MAGCS/Pickles/paperwisemetrics_mag_hybrid5050_df_may26.pickle')
    print("METRICS CALCULATED, time to calculate the means")
    # Get the mean of all the index columns
    # First, drop list columns.
    df = df.drop(['hybrid_recommendations_v2', 'hybrid_binary_v2', 'ground_truth'], axis=1)
    mean_series = df.mean()
    mean_series.to_csv('/home/ashwath/Programs/MAGCS/Evaluation/meanmetrics_mag_june23_hybridv2.tsv', sep='\t', index=True, header=False)
    print("C'est fini.")

    
def calculate_citedbm25_metrics(df):
    """ Calculates metrics at different k (1 to 10 + 20,30,40,50)"""
    #print(df.columns)
    klist = list(range(1, 11))
    klist.extend([20, 30, 40, 50, 100, 200, 300, 500])
    print(klist)
    # 14 x 3 x 4 columns added for each
    for k in tqdm(klist):
        df['average_precision_citedbm25_{}'.format(k)] = df['citedbm25_binary'].progress_apply(lambda x: average_precision(x, k))
        df['recall_citedbm25_{}'.format(k)] = df[['citedbm25_binary', 'ground_truth']].apply(
            lambda x: recall_at_k(x.citedbm25_binary, x.ground_truth, k), axis=1)
        df['reciprocal_rank_citedbm25_{}'.format(k)] = df['citedbm25_binary'].progress_apply(lambda x: reciprocal_rank(x, k))
        df['ndcg_citedbm25_{}'.format(k)] = df['citedbm25_binary'].progress_apply(lambda x: ndcg(x, k))
    #df.to_pickle('/home/ashwath/Programs/MAGCS/Pickles/paperwisemetrics_mag_hybrid5050_df_may26.pickle')
    print("METRICS CALCULATED, time to calculate the means")
    # Get the mean of all the index columns
    # First, drop list columns.
    df = df.drop(['citedbm25_recommendations', 'citedbm25_binary', 'ground_truth'], axis=1)
    mean_series = df.mean()
    mean_series.to_csv('/home/ashwath/Programs/MAGCS/Evaluation/meanmetrics_magcitedbm25_jun18.tsv', sep='\t', index=True, header=False)
    print("C'est fini.")


def calculate_metrics(df):
    """ Calculates metrics at different k (1 to 10 + 20,30,40,50)"""
    #print(df.columns)
    klist = list(range(1, 11))
    klist.extend([20, 30, 40, 50, 100, 200, 300, 500])
    #print(klist)
    # 14 x 3 x 4 columns added for each
    for k in tqdm(klist):
        df['average_precision_hd2v_{}'.format(k)] = df['hd2v_binary'].progress_apply(lambda x: average_precision(x, k))
        df['average_precision_hd2v_wv0dv1_{}'.format(k)] = df['hd2v_wv0dv1_binary'].progress_apply(lambda x: average_precision(x, k))
        #df['average_precision_lda_{}'.format(k)] = df['lda_binary'].progress_apply(lambda x: average_precision(x, k))
        df['average_precision_bm25_{}'.format(k)] = df['bm25_binary'].progress_apply(lambda x: average_precision(x, k))
        df['average_precision_hybrid_{}'.format(k)] = df['hybrid_binary'].progress_apply(lambda x: average_precision(x, k))
        df['recall_hd2v_{}'.format(k)] = df[['hd2v_binary', 'ground_truth']].apply(
            lambda x: recall_at_k(x.hd2v_binary, x.ground_truth, k), axis=1)  
        df['recall_hd2v_wv0dv1_{}'.format(k)] = df[['hd2v_wv0dv1_binary', 'ground_truth']].apply(
            lambda x: recall_at_k(x.hd2v_wv0dv1_binary, x.ground_truth, k), axis=1)  
        #df['recall_lda_{}'.format(k)] = df[['lda_binary', 'ground_truth']].apply(
        #    lambda x: recall_at_k(x.lda_binary, x.ground_truth, k), axis=1)
        df['recall_bm25_{}'.format(k)] = df[['bm25_binary', 'ground_truth']].apply(
            lambda x: recall_at_k(x.bm25_binary, x.ground_truth, k), axis=1)
        df['recall_hybrid_{}'.format(k)] = df[['hybrid_binary', 'ground_truth']].apply(
            lambda x: recall_at_k(x.hybrid_binary, x.ground_truth, k), axis=1)
        df['reciprocal_rank_hd2v_{}'.format(k)] = df['hd2v_binary'].progress_apply(lambda x: reciprocal_rank(x, k))
        df['reciprocal_rank_hd2v_wv0dv1_{}'.format(k)] = df['hd2v_wv0dv1_binary'].progress_apply(lambda x: reciprocal_rank(x, k))
        #df['reciprocal_rank_lda_{}'.format(k)] = df['lda_binary'].progress_apply(lambda x: reciprocal_rank(x, k))
        df['reciprocal_rank_bm25_{}'.format(k)] = df['bm25_binary'].progress_apply(lambda x: reciprocal_rank(x, k))
        df['reciprocal_rank_hybrid_{}'.format(k)] = df['hybrid_binary'].progress_apply(lambda x: reciprocal_rank(x, k))
        df['ndcg_hd2v_{}'.format(k)] = df['hd2v_binary'].progress_apply(lambda x: ndcg(x, k))
        df['ndcg_hd2v_wv0dv1_{}'.format(k)] = df['hd2v_wv0dv1_binary'].progress_apply(lambda x: ndcg(x, k))
        #df['ndcg_lda_{}'.format(k)] = df['lda_binary'].progress_apply(lambda x: ndcg(x, k))
        df['ndcg_bm25_{}'.format(k)] = df['bm25_binary'].progress_apply(lambda x: ndcg(x, k))       
        df['ndcg_hybrid_{}'.format(k)] = df['hybrid_binary'].progress_apply(lambda x: ndcg(x, k))
    #df.to_pickle('/home/ashwath/Programs/MAGCS/Pickles/paperwisemetrics_mag_singlehd2v_df_may20.pickle')
    print("METRICS CALCULATED, time to calculate the means")
    # Get the mean of all the index columns
    # First, drop list columns.
    df = df.drop(['hd2v_recommendations', 'bm25_recommendations', 'hd2v_recommendations_wv0dv1',
                 'hd2v_binary', 'hd2v_wv0dv1_binary', 'bm25_binary', 'hybrid_recommendations', 'hybrid_binary', 'ground_truth'], axis=1)
    mean_series = df.mean()
    mean_series.to_csv('/home/ashwath/Programs/MAGCS/Evaluation/meanmetrics_mag_may29_hd2vboth_bm25_hybrid.tsv', sep='\t', index=True, header=False)
    #mean_series.to_csv('/home/ashwath/Programs/MAGCS/Evaluation/meanmetrics_mag_may20_singlehd2v.tsv', sep='\t', index=True, header=False)
    print("C'est fini.")

if __name__ == '__main__':
    # filename = '/home/ashwath/Programs/MAGCS/mag_testset_contexts.tsv'
    filename = '/home/ashwath/Programs/MAGCS/mag_testset_contexts_grouped.tsv'
    predict(filename)
