import pandas as pd
import re
import contractions
from tqdm import tqdm
import requests
tqdm.pandas()
from gensim.parsing import preprocessing
from metrics import binarize_predictions, precision_at_k, average_precision, recall_at_k, reciprocal_rank, discounted_cumulative_gain, ndcg

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


def solr_recommend(context):
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

filename = '/home/ashwath/Programs/MAGCS/mag_testset_contexts.tsv'
df = pd.read_csv(filename, sep='\t', names=['ground_truth', 'citing_mag_id', 'context'])
df = df[~df.duplicated()]
df['context'] = df['context'].progress_apply(clean_text)
df['wordcount'] = df['context'].apply(lambda x: len(x.split()))
df = df[df.wordcount>8]
df['bm25_recommendations'] = df['context'].progress_apply(solr_recommend)
