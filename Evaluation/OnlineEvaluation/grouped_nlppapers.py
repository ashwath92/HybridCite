import pandas as pd
import re
import contractions
from gensim.parsing import preprocessing
from HyperDoc2Vec import *

hd2vmodel = HyperDoc2Vec.load('/home/ashwath/Programs/MAGCS/MAG-hyperdoc2vec/models/magcsenglish_window20.model')


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

filename='AllNLPcontexts.tsv'
df = pd.read_csv(filename, sep='\t', encoding='utf-8') #17126
df = df[~df.duplicated()] # 17111
# KEEP ONLY CONTEXTS WHICH ARE IN THE TRAINING SET
df = df[df.groundtruth.isin(hd2vmodel.docvecs.offset2doctag)] # 9206
df['normcontext'] = df['context'].apply(clean_text)
contextdict = {normalized: original for normalized, original in df[['normcontext', 'context']].values}
df['wordcount'] = df['normcontext'].apply(lambda x: len(x.split()))
df = df[df.wordcount>8] # 9019
# Remove contexts with less than 9 non-stop words
grouped_df = df.groupby(['magid', 'title', 'normcontext'])['groundtruth'].apply(list).to_frame('ground_truth').reset_index() # 8356

grouped_df['context'] = grouped_df['normcontext'].map(contextdict)
grouped_df = grouped_df.drop('normcontext', axis=1)

# Convert the list to a string with a comma between mag ids (no space)
grouped_df['ground_truth'] = grouped_df['ground_truth'].apply(lambda x: ','.join(str(j) for j in x))
# 8356 left 

# Remove contexts with less than
grouped_df.to_csv('/home/ashwath/Programs/OnlineEvaluation/NLPcontexts_grouped.tsv', sep='\t', encoding='utf-8', index=False)

print(grouped_df.shape) # 
