import numpy as np
import contractions
from nltk.stem import SnowballStemmer

#try:
#    import modin.pandas as pd
#except ModuleNotFoundError:
import pandas as pd
from time import sleep, time
from tqdm import tqdm
tqdm.pandas()
import pickle
from gensim.parsing import preprocessing
from gensim import matutils
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet
# If mallet doesn't work, use normal LDA.
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities

mallet_path = '/home/ashwath/mallet-2.0.8/bin/mallet'


# LOAD MODELS
loadmodstart = time()
id2word_dictionary = corpora.Dictionary.load('/home/ashwath/Programs/MAGCS/LDA/mag.dict')
corpus = corpora.MmCorpus('/home/ashwath/Programs/MAGCS/LDA/mag_bow_corpus.mm')
#try:
#   ldamallet = LdaMallet.load('/home/ashwath/Programs/MAGCS/LDA/ldamallet_mag.model')
#    vec_bow_test = id2word_dictionary.doc2bow(['test'])
#    vec_ldamallet = ldamallet[vec_bow_test]
#except subprocess.CalledProcessError:
#    print("LDA MALLET COULDN'T READ INSTANCE FILE. USING NORMAL LDA INSTEAD")
ldamallet = LdaModel.load('/home/ashwath/Programs/MAGCS/LDA/lda_mag.model')
    
malletindex = similarities.MatrixSimilarity.load('/home/ashwath/Programs/MAGCS/LDA/simIndexMag.index')
with open('/home/ashwath/Programs/MAGCS/LDA/docid_to_magid_training_mag.pickle', 'rb') as pick:
    docid_to_magid = pickle.load(pick)


def binarize_predictions(relevant_list, predicted_list):
    #print(predicted_list)
    """Returns 2 if the first entry is present in the predictions, 1 if one of the
    other relevant items is present in the predictions, 0 otherwise."""
    #bin_list = []
    #for ground_truth, pred in zip(relevant_list, predicted_list):
    return [2 if entry == relevant_list[0] else 1 if entry in relevant_list[1:] else 0 
            for entry in predicted_list]

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
    sleep(0.2)
    vec_bow = id2word_dictionary.doc2bow(context_list)
    # This line takes a LONG time: it has to map to each of the 300 topics
    vec_ldamallet = ldamallet[vec_bow]
    # Convert the query to LDA space
    sims = malletindex[vec_ldamallet]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])[:topn]
    # sims is a list of tuples of (docid -- line num in original training file, probability)
    return [docid_to_magid.get(docid) for docid, prob in sims] 

def calculate_metrics(df):
    """ Calculates metrics at different k (1 to 10 + 20,30,40,50)"""
    #print(df.columns)
    klist = list(range(1, 10))
    klist.extend([20, 30, 40, 50, 100, 200, 300, 500])
    print(klist)
    # 14 x 3 x 4 columns added for each
    for k in tqdm(klist):
        #df['average_precision_hd2v_{}'.format(k)] = df['hd2v_binary'].progress_apply(lambda x: average_precision(x, k))
        df['average_precision_lda_{}'.format(k)] = df['lda_binary'].progress_apply(lambda x: average_precision(x, k))
        #df['average_precision_bm25_{}'.format(k)] = df['bm25_binary'].progress_apply(lambda x: average_precision(x, k))
        #print(df)
        #df['recall_hd2v_{}'.format(k)] = df[['hd2v_binary', 'ground_truth']].apply(
        #    lambda x: recall_at_k(x.hd2v_binary, x.ground_truth, k), axis=1)    
        df['recall_lda_{}'.format(k)] = df[['lda_binary', 'ground_truth']].apply(
            lambda x: recall_at_k(x.lda_binary, x.ground_truth, k), axis=1)
        #df['recall_bm25_{}'.format(k)] = df[['bm25_binary', 'ground_truth']].apply(
        #    lambda x: recall_at_k(x.bm25_binary, x.ground_truth, k), axis=1)
        #df['reciprocal_rank_hd2v_{}'.format(k)] = df['hd2v_binary'].progress_apply(lambda x: reciprocal_rank(x, k))
        df['reciprocal_rank_lda_{}'.format(k)] = df['lda_binary'].progress_apply(lambda x: reciprocal_rank(x, k))
        #df['reciprocal_rank_bm25_{}'.format(k)] = df['bm25_binary'].progress_apply(lambda x: reciprocal_rank(x, k))
        #df['ndcg_hd2v_{}'.format(k)] = df['hd2v_binary'].progress_apply(lambda x: ndcg(x, k))
        df['ndcg_lda_{}'.format(k)] = df['lda_binary'].progress_apply(lambda x: ndcg(x, k))
        #df['ndcg_bm25_{}'.format(k)] = df['bm25_binary'].progress_apply(lambda x: ndcg(x, k))
    df.to_csv('/home/ashwath/Programs/MAGCS/Evaluation/paperwisemetrics_magLDA_3models_500.tsv', sep='\t')
    df.to_pickle('/home/ashwath/Programs/MAGCS/Pickles/paperwisemetrics_magLDA_3models_df.pickle')
    print("METRICS CALCULATED, time to calculate the means")
    # Get the mean of all the index columns
    # First, drop list columns.
    df = df.drop(['lda_recommendations', 'lda_binary', 'ground_truth'], axis=1)
    mean_series = df.mean()
    mean_series.to_csv('/home/ashwath/Programs/MAGCS/Evaluation/meanmetrics_mag_lda.tsv', sep='\t', index=True, header=False)
    print("C'est fini.")

#df = pd.read_pickle('/home/ashwath/Programs/MAGCS/MAG-hyperdoc2vec/recommendationsids.pickle')

#df['lda_recommendations'] = df['context_for_lda'].progress_apply(lda_recommend)
#print('lda recommendations done')
#df['lda_binary'] = df[['ground_truth', 'lda_recommendations']].apply(
#    lambda x: binarize_predictions(x.ground_truth, x.lda_recommendations), axis=1)
sleep(0.3)
#print("Binarized")

#df.to_pickle('/home/ashwath/Programs/MAGCS/Pickles/ldarecommendations_mag_3models_500_df.pickle')
df = pd.read_pickle('/home/ashwath/Programs/MAGCS/Pickles/ldarecommendations_mag_3models_500_df.pickle')
calculate_metrics(df[['lda_recommendations', 'lda_binary', 'ground_truth']])

