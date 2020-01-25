import pandas as pd
import numpy as np

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
    #print('predictedbinlist unreduced=', predicted_bin_list)
    predicted_bin_array = np.array(predicted_bin_list)
    predicted_bin_array = predicted_bin_array[:k]
    # TAKE CARE: np.log2(1) = 0, we want it to be 1.
    # Element-wise division, first term is not divided by log 1, but by 1 instead.
    # it is a cumulative sum
    #print("k=", k, '!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #print(predicted_bin_array, 'pred bin array')
    #print('k+1', k+1, 'arange',  np.log2(np.arange(2, k+1)))
    try:
        return predicted_bin_array[0] + np.sum(predicted_bin_array[1:] / np.log2(np.arange(2, k+1)))
    except ValueError as v:
        print(predicted_bin_array)
        print(np.log2(np.arange(2, k+1)))
        return 0
        
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
