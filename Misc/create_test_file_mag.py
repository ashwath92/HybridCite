import pickle
import csv
from gensim.parsing import preprocessing
import re
import os
from tqdm import tqdm
import psycopg2
import psycopg2.extras
conn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=shetland.informatik.uni-freiburg.de")
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
query = """

     SELECT englishcspapers.paperid, englishcspapers.papertitle, citationcontexts.citationcontext, citationcontexts.paperreferenceid from
        (
            SELECT papertitle, computersciencepapers.paperid from 
            (
                SELECT papers.paperid, papertitle FROM papers 
                 INNER JOIN 
                (SELECT paperid from paperfieldsofstudy WHERE fieldofstudyid=41008148) AS fieldsofstudy 
                 ON papers.paperid=fieldsofstudy.paperid where papers.publishedyear in (2018,2019)
            ) AS computersciencepapers
            INNER JOIN 
            (SELECT paperid FROM paperlanguages WHERE languagecode='en') AS languages 
            ON languages.paperid=computersciencepapers.paperid
        ) AS englishcspapers INNER JOIN 
         
    (SELECT paperid, paperreferenceid, citationcontext
    FROM papercitationcontexts) AS citationcontexts 
    ON citationcontexts.paperid=englishcspapers.paperid; """

cur.execute(query)

# 15668 papers
with open('testmagids.pickle','rb') as file:
    testmagids = pickle.load(file)

with open('trainingmagids.pickle','rb') as pfile:
    allmagpapers_set = pickle.load(pfile)


with open('/home/ashwath/Programs/MAGCS/mag_testset_contexts.tsv', 'w') as outfile:
    #fieldnames = ['cited_mag_id', 'context']
    #writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=fieldnames)
    for row in cur:
        #parts = line.split('\u241E')
        #print(i)
        citingmagid = row.get('paperid')
        if citingmagid not in testmagids:
            # This was not a test set paper id. Skip
            continue
        citedmagid = repr(row.get('paperreferenceid'))
        # Check if the paper id is in the training set. If not, discard the context, it cannot be predicted
        # This is the case for the main mag id
        if citedmagid not in allmagpapers_set:
            print(citedmagid)
            continue
        context = row.get('citationcontext').replace('\n', '')
                
        # Create a comma-separated string out of the list
        # Remove the citation sybmols
        #context = preprocessing.strip_multiple_whitespaces(parts[3].replace('MAINCIT', ' ').replace('CIT', '').replace(' ,', ''))
        # Don't clean the context now. Leave that to each individual testing algorithm
        #writer.writerow({'cited_mag_id': citedmagid, 'context': context})
        outfile.write('{}\t{}\t{}\n'.format(citedmagid, citingmagid, context)) 