import pickle
import pandas as pd
import csv
from gensim.parsing import preprocessing
import re
import sqlite3
import os

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

# mag arxiv mapping db connection
sconn = db_connect()
#scur = sconn.cursor()

with open('Pickles/allmagpapers_en_magcontexts.pickle', 'rb') as pick:
    allmagpapers_set = pickle.load(pick)

# Get the test set ARXIV ids from a saved test set tsv. 2 columns: arxiv_id, mag_id
# Get the test set: we just need the arxiv id
testsetquery = """select arxiv_id, mag_id 
                     from arxivcs_mag 
                     where arxiv_id like '17%'
                     group by mag_id;

               """
# shape: (18642, 2)
testresdf = pd.read_sql_query(testsetquery, sconn)
testresdf.arxiv_id = testresdf.arxiv_id.astype('str')
# Save it
testresdf.to_csv('AdditionalOutputs/test_ids.tsv', index=False, sep='\t')

testsetarxivids = set(testresdf.arxiv_id.tolist())

#print(testsetarxivids)
#del testresdf

with open('/home/ashwath/Files/items_CSall_1s_5mindoc_5mincont.csv', 'r') as infile,\
     open('/home/ashwath/Programs/ArxivCS/arxiv_testset_contexts.tsv', 'w') as outfile:
    #fieldnames = ['cited_mag_id', 'context']
    #writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=fieldnames)
    for i, line in enumerate(infile):
        parts = line.split('\u241E')
        print(i)
        citedmagids = [parts[0]]
        # Additional cited mag ids are separtated by a field separator, 
        additionalmagids = parts[1].split('\u241F')
        # If there are adjacent mag ids, they are secondary ground truths
        if additionalmagids != ['']:
            citedmagids.extend(additionalmagids)
        # parts[1] is a space
        citingarxivid = parts[2]
        if citingarxivid not in testsetarxivids:
            # This was not a test set paper id. Skip
            continue
        print(citedmagids, citingarxivid)
        # Check if the paper id is in the training set. If not, discard the context, it cannot be predicted
        # This is the case for the main mag id
        if citedmagids[0] not in allmagpapers_set:
            continue
        # For the secondary mag ids, it is a different case. If they are not present in the training set,
        # just remove them from the ground truth list.
        removemagidslist = []
        for citedmagindex, citedmagid in enumerate(citedmagids[1:]):
            # Check in turn if each of the secondary cited mag ids are in the TRAINING SET.
            if citedmagid not in allmagpapers_set:
                removemagidslist.append(citedmagid)
                
        # removemagidslist may be empty at this stage, or have multiple mag ids.
        # Remove the elements which are present in removedmagidsdlist from the citedmagids list 
        # Leave the first element of the list: citedmagids[0] as is.
        citedmagids[1:] = [magid for magid in citedmagids[1:] if magid not in removemagidslist]
        citedmagids_string = ','.join(citedmagids)
        # Create a comma-separated string out of the list
        # Remove the citation sybmols
        context = preprocessing.strip_multiple_whitespaces(parts[3].replace('MAINCIT', ' ').replace('CIT', '').replace(' ,', ''))
        # Don't clean the context now. Leave that to each individual testing algorithm
        #writer.writerow({'cited_mag_id': citedmagid, 'context': context})
        outfile.write('{}\t{}\t{}\n'.format(citedmagids_string, citingarxivid, context))