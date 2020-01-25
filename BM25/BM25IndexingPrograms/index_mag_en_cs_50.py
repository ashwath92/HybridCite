from gensim.parsing import preprocessing
import contractions
from tqdm import tqdm
import pysolr

# IMPORTANT: I'M KEEPING THE REFERENCE IDS IN THE CONTEXTS. SO WHILE CHECKING BM25,
# CONTEXTS WHICH REFER TO THE SAME PAPER MIGHT BE MORE SIMILAR (IF CITATIONS ALREADY
#EXIST)

def clean_text(text):
    """ Cleans the text in the only argument in various steps 
    ARGUMENTS: text: content/title, string
    RETURNS: cleaned text, string"""
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
    return text

solr = pysolr.Solr('http://localhost:8983/solr/mag_en_cs_50', always_commit=True)
list_for_solr = []
rownum = 0
filename = '/home/ashwath/Programs/MAGCS/MAG-hyperdoc2vec/input/mag_training_data_50citationsmin.txt'
with open(filename, 'r') as file:
    # list of lists
    for line in tqdm(file):
        solr_record = dict()
        rownum += 1
        parts = clean_text(line).split()
        paperid = parts[0]
        content = ' '.join(parts[1:])
    
        solr_record['paperid'] = paperid
        solr_record['content'] = content
        if rownum % 10000 == 0:
            list_for_solr.append(solr_record)
            solr.add(list_for_solr)
            list_for_solr = []
            print(rownum)
        else:
            list_for_solr.append(solr_record)
    solr.add(list_for_solr)
