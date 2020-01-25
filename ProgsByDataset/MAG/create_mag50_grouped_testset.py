import pandas as pd
import re
import contractions
from gensim.parsing import preprocessing

filename = '/home/ashwath/Programs/MAGCS/mag50_testset_contexts.tsv'

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


df = pd.read_csv(filename, sep='\t', names=['ground_truth', 'citing_mag_id', 'context'], encoding='utf-8')
df = df[~df.duplicated()]
df['normcontext'] = df['context'].apply(clean_text)
contextdict = {normalized: original for normalized, original in df[['normcontext', 'context']].values}

grouped_df = df.groupby(['citing_mag_id', 'normcontext'])['ground_truth'].apply(list).to_frame('ground_truth').reset_index() # 1523

grouped_df['context'] = grouped_df['normcontext'].map(contextdict)
grouped_df = grouped_df.drop('normcontext', axis=1)
# Reorder the columns
grouped_df = grouped_df[['ground_truth', 'citing_mag_id', 'context']]

# Convert the list to a string with a comma between mag ids (no space)
grouped_df['ground_truth'] = grouped_df['ground_truth'].apply(lambda x: ','.join(str(j) for j in x))

grouped_df.to_csv('/home/ashwath/Programs/MAGCS/mag50_testset_contexts_grouped.tsv', sep='\t', encoding='utf-8', index=False, header=None)

print(grouped_df.shape) # 

#df['hd2v_recommendations'] = df['context'].progress_apply(hd2v_recommend)
#df['hd2v_recommendations_wv0dv1'] = df['context'].progress_apply(hd2v_wvindvout_recommend)

