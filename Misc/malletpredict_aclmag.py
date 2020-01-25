import pickle
from time import sleep
from gensim.models.wrappers import LdaMallet
try:
    import modin.pandas as pd
except ModuleNotFoundError:
    import pandas as pd

from gensim import corpora, models, similarities
from tqdm import tqdm
tqdm.pandas()

def calculate_metrics(df):
    """ Calculates metrics at different k (1 to 10 + 20,30,40,50)"""
    #print(df.columns)
    klist = list(range(1, 10))
    klist.extend([20, 30, 40, 50, 100, 200, 300, 500])
    print(klist)
    # 14 x 3 x 4 columns added for each
    for k in tqdm(klist):
        df['average_precision_lda_{}'.format(k)] = df['lda_binary'].apply(lambda x: average_precision(x, k))
        #print(df)
        df['recall_lda_{}'.format(k)] = df[['lda_binary', 'ground_truth']].apply(
            lambda x: recall_at_k(x.lda_binary, x.ground_truth, k), axis=1)
        df['reciprocal_rank_lda_{}'.format(k)] = df['lda_binary'].apply(lambda x: reciprocal_rank(x, k))
        df['ndcg_lda_{}'.format(k)] = df['lda_binary'].apply(lambda x: ndcg(x, k))
    df.to_csv('/home/ashwath/Programs/ACLAAn/Evaluation/malletmetrics_aclmag_3models_500.tsv', sep='\t')
    df.to_pickle('/home/ashwath/Programs/ACLAAn/Pickles/malletmetrics_aclmag_3models_df.pickle')
    print("METRICS CALCULATED, time to calculate the means")
    # Get the mean of all the index columns
    mean_series = df.mean()
    mean_series.to_csv('/home/ashwath/Programs/ACLAAn/Evaluation/malletmetrics.tsv', sep='\t', index=True, header=False)
    print("C'est fini.")


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


def lda_recommend(context_list):
    """ With multiprocessing using Dask"""
    print("Recommending")
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

#df = pd.read_pickle('recommendationsids.pickle')
df = pd.read_pickle('/home/ashwath/Programs/ACLAAn/Pickles/recommendations_aclmag_3models_500_df.pickle')
malletindex = similarities.MatrixSimilarity.load('/home/ashwath/Programs/ACLAAn/LDA/simIndexAcl.index')
with open('/home/ashwath/Programs/ACLAAn/LDA/docid_to_magid_training_acl.pickle', 'rb') as pick:
    docid_to_magid = pickle.load(pick)
id2word_dictionary = corpora.Dictionary.load('/home/ashwath/Programs/ACLAAn/LDA/aclmag.dict')
corpus = corpora.MmCorpus('/home/ashwath/Programs/ACLAAn/LDA/aclmag_bow_corpus.mm')

ldamallet = LdaMallet.load('/home/ashwath/Programs/ACLAAn/LDA/lda_model.model')

df['lda_recommendations'] = df['context_for_lda'].progress_apply(lda_recommend)
df['lda_binary'] = df[['ground_truth', 'lda_recommendations']].apply(
        lambda x: binarize_predictions(x.ground_truth, x.lda_recommendations), axis=1)
df.to_pickle('/home/ashwath/Programs/ACLAAn/Pickles/malletrecommendations_aclmag_500_df.pickle')
df.to_csv('/home/ashwath/Programs/ACLAAn/Evaluation/malletrecommendations_aclmag_500.tsv', sep='\t')
sleep(0.3)
calculate_metrics(df[['lda_recommendations', 'lda_binary', 'ground_truth']])