""" This prepares a file for hyperdoc2vec from the arxiv full text (only papers mapped to mag ids). """
import os
import re
import csv
import pickle
import sqlite3
import psycopg2
import psycopg2.extras
from gensim.parsing import preprocessing
from gensim.utils import to_unicode
import pandas as pd
from tqdm import tqdm

basepath = '/home/ashwath/Programs'
dbpath = os.path.join(basepath, 'ArxivCS', 'SQLITEDB', 'arxivcs_mag_mapping.sqlite3')
def db_connect(set_params=False, path = dbpath):
    """ Connects to sqlite3 db given via a parameter/uses a default parameter.
    It sets timeout=10 to prevent sqlite getting locked in between inserts. It 
    also sets detect_types to allow datetime/date datatypes in tables. """
    connection = sqlite3.connect(path, timeout=10, 
                                 detect_types=sqlite3.PARSE_DECLTYPES)
    if set_params is True:
        # Speed up insertions: only called while creating the database
        connection.execute('PRAGMA main.journal_mode=WAL;')
        connection.execute('PRAGMA main.cache_size=10000;')
        connection.execute('PRAGMA main.locking_mode=EXCLUSIVE;')   
    return connection

# GLOBALS
# Hyperdoc2vec markers for citations
docid_prefix='=-='
docid_suffix='-=-'
# IMPORTANT: I need a set of mag ids which are cited so that.
allmagpaperids = set()
# mag arxiv mapping db connection
sconn = db_connect()
scur = sconn.cursor()

# context connection: for getting the mag id of the CITED papers 
meta_db_path = '/vol2/unarXive/arxiv-txt-data/metadata.db'
cconn = db_connect(path=meta_db_path)
ccur = cconn.cursor()

# Some arxiv ids are mapped to 2 magids, keep only 1 (data problem)
# 72246 rows in the results (out of 72315): 69 duplicates

# Training set is all years until 2016 (2017 is the test set)
# Training set: 62296 papers
# Test set: 9954 papers
trainingquery = """select arxiv_id, mag_id 
                     from arxivcs_mag 
                     where arxiv_id not like '17%'
                     group by mag_id;

                  """
# Write test set
testsetquery = """select arxiv_id, mag_id 
                     from arxivcs_mag 
                     where arxiv_id like '17%'
                     group by mag_id;

                  """
testresdf = pd.read_sql_query(testsetquery, sconn)
testresdf.to_csv('AdditionalOutputs/test_ids.tsv', index=False, sep='\t')

trainresdf = pd.read_sql_query(trainingquery, sconn)
trainresdf.to_csv('AdditionalOutputs/training_ids.tsv', index=False, sep='\t')
# Get a Series of mag ids for which we have full text
mag_id_series = trainresdf['mag_id']
# IMP: There seems to be some problem with the data?? Multiple arxiv ids are mapped to the same mag id
# Doing select mag_id from arxivcs_mag, and read_sql_query, then
# df[df.isin(df[df.duplicated()])] gives 69 records.
# Get a set of mag ids (mapped from arxiv of course) which have full text
inarxiv_papers_set = set(mag_id_series.tolist())

# POSTGRES connection obj and cursor
pconn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=shetland.informatik.uni-freiburg.de")
pcur = pconn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# POSTGRES QUERY
magonly_query = """
SELECT titleandabstract.paperid, papertitle, abstract, contexts, referenceids
FROM  
    (
        SELECT papers.paperid, papertitle, abstract FROM papers INNER JOIN paperabstracts 
        ON papers.paperid=paperabstracts.paperid
        WHERE papers.paperid=%s) AS titleandabstract INNER JOIN 
        (
            SELECT paperid, string_agg(paperreferenceid::character varying, ',') AS referenceids,
            string_agg(citationcontext, ' ||--|| ') AS contexts 
            FROM papercitationcontexts 
            WHERE paperid=%s 
            GROUP BY paperid
        ) AS listofcontexts
        ON titleandabstract.paperid=listofcontexts.paperid;"""

# Arxiv citing, cited list for Tarek. 
arxiv_citing_cited_file = open('AdditionalOutputs/citing_cited.tsv', 'w')
fieldnames = ['citing_arxiv_id', 'citing_mag_id', 'cited_mag_id']
writer = csv.DictWriter(arxiv_citing_cited_file, delimiter="\t", fieldnames=fieldnames)
writer.writeheader()

citation_pattern = re.compile(r'(\{\{cite:)([a-zA-z0-9-]+)(\}\})')

def get_mag_from_uuid(matchobject):
    """ """
    cited_uuid = matchobject.group(2)
    uuidquery = "select mag_id from bibitemmagidmap where uuid='{}'".format(cited_uuid)
    ccur.execute(uuidquery)
    res = ccur.fetchone()
    if res is None:
        # If the uuid does not map to a mag id, replace with the word citation.
        #wordindex_magid_dict[i] = 'citation'
        return 'citation'
    else:
        fetched_magid = res[0]
        allmagpaperids.add(fetched_magid)
        return '{}{}{}'.format(docid_prefix, fetched_magid, docid_suffix)

""" def read_arxiv_addmagids(arxiv_id, mag_id, ccur):
    Read arxiv full text,
    arxiv_filepath = '/vol2/unarXive/arxiv-txt-data/{}.txt'.format(arxiv_id)
    with open(arxiv_filepath, 'r') as arxivfile:
        content = arxivfile.read()
    words = content.split()
    # Replace all {{cite:ac7d7c84-d6e0-461d-a1fc-36f7ee323c07}}, i.e. \{\}cite:.*\}\}
    # Get all the word indices which need to be replaced and put it in a dict with 
    # the corresponding mag id from the db.
    wordindex_magid_dict = dict()
    for i, word in enumerate(words):
        if word.startswith('{{cite'):
            # Skip first 7 characters, uuid is the part between {{cite and }}
            uuid = word[7:word.find('}}')]
            #print(uuid)
            # Use uuid to get the mag id from the contexts db
            query = "select mag_id from bibitemmagidmap where uuid='{}'".format(uuid)
            ccur.execute(query)
            # Res is a tuple of one element E.g.: ('2133990480',)
            res = ccur.fetchone()
            if res is None:
                # If the uuid does not map to a mag id, replace with the word citation.
                wordindex_magid_dict[i] = 'citation'
            else:
                fetched_magid = res[0]
                # Add to the seen set
                allmagpaperids.add(fetched_magid)
                writer.writerow({'citing_arxiv_id': arxiv_id, 'citing_mag_id': mag_id,
                                 'cited_mag_id': fetched_magid})
                wordindex_magid_dict[i] = '{}{}{}'.format(docid_prefix, res[0], docid_suffix)
    # Do the replacements in the words list
    for index, replacement_citation in wordindex_magid_dict.items():
        words[index] = replacement_citation
    # Convert the words list back to a string
    # Make sure to add the citing paper mag id as the first word in the line
    content = '{} {}\n'.format(mag_id, ' '.join(words))
    return content
"""

def read_arxiv_addmagids(arxiv_id, mag_id, ccur):
    """ Read arxiv full text, replace citations with mag id """
    arxiv_filepath = '/vol2/unarXive/arxiv-txt-data/{}.txt'.format(arxiv_id)
    with open(arxiv_filepath, 'r') as arxivfile:
        content = arxivfile.read().replace('\n', ' ')
    # Replace all {{cite:ac7d7c84-d6e0-461d-a1fc-36f7ee323c07}}, i.e. \{\}cite:.*\}\}
    # Get all the word indices which need to be replaced and put it in a dict with 
    # the corresponding mag id from the db.
    # Do the replacements in the words list
    content = citation_pattern.sub(get_mag_from_uuid, content)

    # Convert the words list back to a string
    # Make sure to add the citing paper mag id as the first word in the line
    content = '{} {}\n'.format(mag_id, content)
    return content


def clean_text(text):
    """ Cleans the text in the only argument in various steps 
    ARGUMENTS: text: content/title, string
    RETURNS: cleaned text, string"""
    # Replace newlines by space. We want only one doc vector.
    text = text.replace('\n', ' ').lower()
    # Remove URLs
    #text = re.sub(r"http\S+", "", text)
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Remove stop words
    #text = preprocessing.remove_stopwords(text)
    
    #text = preprocessing.strip_tags(text)
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
    return text


def add_additional_papers():
    """ Add additional papers for which full text from Arxiv is not present. Care is taken that while
    adding references to THESE papers, these references should be in the set of papers stored
    in the allmagpaperids set (otherwise, there will be additional papers in the reference part
    of the concat contexts which are not in the files in the text.
    ALSO NOTE that allmagpaperids contains all papers which either cite or are cited so far
    inarxiv_papers_set contains the set of papers which are in arxiv (citing)
    A set difference (allmagpaperids - inarxiv_papers_set) gives the set of mag_ids for which we 
    get additional text"""
    with open('Pickles/inarxiv_papers_set.pickle', 'wb') as picc:
        pickle.dump(inarxiv_papers_set, picc)
    with open('Pickles/allmagpapers_en_magcontexts.pickle', 'wb') as picc2:
        pickle.dump(allmagpaperids, picc2)
    additional_mag_ids = allmagpaperids - inarxiv_papers_set
    for paperid in tqdm(additional_mag_ids):
        pcur.execute(magonly_query, (paperid, paperid))
        # Get paperid, contexts, abstract, title, refids of current paper id
        for row in pcur:
            # row is a dict with keys:
            # dict_keys(['paperid', 'papertitle', 'abstract', 'contexts', 'referenceids'])
            paperid = row.get('paperid')
            # Get all contexts and reference ids (delimiters set in the pSQL query)
            contexts = row.get('contexts').replace('\n', ' ')
            referenceids = row.get('referenceids')
            title = clean_text(row.get('papertitle'))
            abstract = clean_text(row.get('abstract'))
            print(title)
            # Get a single string for all the contexts
            if contexts is not None and referenceids is not None:
                contexts = contexts.split(' ||--|| ')
                referenceids = referenceids.split(',')
                contexts_with_refs = []
                # Go through context, refid pairs, one at a time
                for context, referenceid in zip(contexts, referenceids):
                    # VERY VERY IMPORTANT: check if the referenceid is not present in the allmagpaperids set,
                    # IGNORE IT! DESIGN DECISION: the other choice is to have a LOT of passes. 
                    if referenceid in allmagpaperids:
                        contextlist = clean_text(context).split()
                        # Insert the reference id as the MIDDLE word of the context
                        # NOTE, when multiple reference ids are present, only 1 is inserted. Mag issue.
                        # In the eg. nips file, it's like this: this paper uses our previous work on weight space 
                        # probabilities =-=nips05_0451-=- =-=nips05_0507-=-. 
                        index_to_insert = len(contextlist) // 2
                        value_to_insert = docid_prefix + referenceid + docid_suffix
                        # Add the ref id with the prefix and suffix
                        contextlist.insert(index_to_insert, value_to_insert)
                        # Insert the context with ref id into the contexts_with_refs list
                        contexts_with_refs.append(' '.join(contextlist))
                    # else: do nothing, next iteration
                # After all the contexts azre iterated to, make them a string.
                contexts_concatenated = ' '.join(contexts_with_refs)                    
            else:
                contexts_concatenated = ''
                # Do not write these to file????? OR 
            # Concatenate the paperid, title, abstract and the contexts together.
            content = "{} {} {} {}\n".format(paperid, title, abstract, contexts_concatenated)
            content = to_unicode(content)
            # Get rid of empty files
            #parts = content.split()
            #parts = [ word for word in parts if not all(letter in string.punctuation for letter in word)]
            #print(parts)
            #content = ' '.join(parts)
            if content.strip() != '':
                fulltext_file.write(content)
                print("Written file for {}".format(paperid))

def main():
    """ """
    
    scur.execute(trainingquery)
    with open('arxiv_hd2v_training.txt', 'w') as file:
        for row in tqdm(scur):
            # Each row is a tuple. E.g.: ('0704.0047', '2042449614')
            # Open the file
            arxiv_id = row[0]
            mag_id = row[1]
            allmagpaperids.add(mag_id)
            content = read_arxiv_addmagids(arxiv_id, mag_id, ccur)
            file.write(content)

    # Add additional content : abstact + title + concatenated contexts from MAG
    # Note that the citation marker (cited paper id) is always placed bang in the centre
    # of the context.
    add_additional_papers()

    # Close files and db connections
    arxiv_citing_cited_file.close()
    sconn.close()
    cconn.close()
    pconn.close()

if __name__ == '__main__':
    main()