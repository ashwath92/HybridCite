
import re
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from gensim import matutils
from gensim.parsing import preprocessing
import gensim
import contractions
from tqdm import tqdm
tqdm.pandas()
from time import time, sleep

from metrics import binarize_predictions, precision_at_k, average_precision, recall_at_k, reciprocal_rank, discounted_cumulative_gain, ndcg

# papervecs is already a keyedvector, but note that it also has a wv attribute.
papervecs = KeyedVectors.load_word2vec_format('Paper2Vec_mag50.txt', binary=False)

# load doc2vec dm model
model = gensim.models.Doc2Vec.load('MAG50CompScienced2v.dat')

def calculate_metrics(df):
    """ Calculates metrics at different k (1 to 10 + 20,30,40,50)"""
    #print(df.columns)
    klist = list(range(1, 11))
    klist.extend([20, 30, 40, 50, 100, 200, 300, 500])
    print(klist)
    # 14 x 3 x 4 columns added for each
    for k in tqdm(klist):
        df['average_precision_p2v_{}'.format(k)] = df['p2v_binary'].apply(lambda x: average_precision(x, k))
        df['recall_p2v_{}'.format(k)] = df[['p2v_binary', 'ground_truth']].apply(
            lambda x: recall_at_k(x.p2v_binary, x.ground_truth, k), axis=1)    
        df['reciprocal_rank_p2v_{}'.format(k)] = df['p2v_binary'].apply(lambda x: reciprocal_rank(x, k))
        df['ndcg_p2v_{}'.format(k)] = df['p2v_binary'].apply(lambda x: ndcg(x, k))
        df['average_precision_d2v_{}'.format(k)] = df['d2v_binary'].apply(lambda x: average_precision(x, k))
        df['recall_d2v_{}'.format(k)] = df[['d2v_binary', 'ground_truth']].apply(
            lambda x: recall_at_k(x.d2v_binary, x.ground_truth, k), axis=1)    
        df['reciprocal_rank_D2v_{}'.format(k)] = df['d2v_binary'].apply(lambda x: reciprocal_rank(x, k))
        df['ndcg_d2v_{}'.format(k)] = df['d2v_binary'].apply(lambda x: ndcg(x, k))
    
    df.to_pickle('/home/ashwath/Programs/MAGCS/Pickles/paperwisemetrics_mag50_d2v_p2v_may23_df.pickle')
    print("METRICS CALCULATED, time to calculate the means")
    # Get the mean of all the index columns
    df = df.drop(['p2v_recommendations', 'p2v_binary', 'd2v_recommendations', 'd2v_binary', 'ground_truth'], axis=1)
    mean_series = df.mean()
    mean_series.to_csv('/home/ashwath/Programs/MAGCS/Evaluation/meanmetrics_mag50_d2v_p2v_may21.tsv', sep='\t', index=True, header=False)
    print("C'est fini.")


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

def paper2vec_recommend(context):
    """ Make recommendations based on the Paper2vec vectors."""
    #if not hasattr(papervecs, 'syn0'):
    #    raise RuntimeError("Parameters required for predicting the output words not found.")
    topn = 500
    context_words_list = context.split()
    sleep(0.3)
    # REMEMBER: Here, papervecs.wv.vocab contains not words, but docids
    # Use the doc2vec wv 
    word_vocabs = [model.wv.vocab[w] for w in context_words_list if w in model.wv.vocab]
    word2_indices = [word.index for word in word_vocabs]
    l1 = np.sum(model.wv.syn0[word2_indices], axis=0)
    if word2_indices:
        l1 /= len(word2_indices)
    prob_values = np.exp(np.dot(l1, papervecs.syn0.T))
    #prob_values = np.exp(np.dot(l1, model.docvecs.doctag_syn0.T))
    prob_values = np.nan_to_num(prob_values)
    prob_values /= sum(prob_values)
    # some of the vectors in papervecs stand for docs, some just for words (where are these ids coming from?)
    top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
    return [papervecs.index2entity[index1] for index1 in top_indices]
    #return [model.docvecs.offset2doctag[index1] for index1 in top_indices]

def doc2vec_recommend(context):
    """ Make recommendations based on the Paper2vec vectors."""
    topn = 500
    context_words_list = context.split()
    #sleep(0.1)
    # Use the doc2vec wv 
    word_vocabs = [model.wv.vocab[w] for w in context_words_list if w in model.wv.vocab]
    word2_indices = [word.index for word in word_vocabs]
    l1 = np.sum(model.wv.syn0[word2_indices], axis=0)
    if word2_indices:
        l1 /= len(word2_indices)
    prob_values = np.exp(np.dot(l1, model.docvecs.doctag_syn0.T))
    prob_values = np.nan_to_num(prob_values)
    prob_values /= sum(prob_values)
    top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
    return [model.docvecs.offset2doctag[index1] for index1 in top_indices]


filename = '/home/ashwath/Programs/MAGCS/mag50_testset_contexts.tsv'
df = pd.read_csv(filename, sep='\t', names=['ground_truth', 'citing_mag_id', 'context'])
print('read file')
# Remove duplicates.
# 3220 rows (50 duplicates)
df = df[~df.duplicated()]
# Keep only those which have ground truths corresponding to a Paper2vec training vector
df = df[df.ground_truth.isin(papervecs.index2entity)]
print('removed duplicates')
df['ground_truth'] = df['ground_truth'].astype('str').apply(lambda x: x.split(','))
df['context'] = df['context'].apply(clean_text)
df['wordcount'] = df['context'].apply(lambda x: len(x.split()))
df = df[df.wordcount>8]
print('ready for recommending')
df['p2v_recommendations'] = df['context'].progress_apply(paper2vec_recommend)
df['d2v_recommendations'] = df['context'].progress_apply(doc2vec_recommend)
print('recommendations done')
df['p2v_binary'] = df[['ground_truth', 'p2v_recommendations']].apply(
        lambda x: binarize_predictions(x.ground_truth, x.p2v_recommendations), axis=1)
df['d2v_binary'] = df[['ground_truth', 'd2v_recommendations']].apply(
        lambda x: binarize_predictions(x.ground_truth, x.d2v_recommendations), axis=1)

print('binarized recommendations')
df.to_pickle('/home/ashwath/Programs/MAGCS/Pickles/recommendations_mag50_p2v_d2v_may22.pickle')
print('pickled recommendations')
calculate_metrics(df[['p2v_recommendations', 'p2v_binary', 'd2v_recommendations', 'd2v_binary', 'ground_truth']])

