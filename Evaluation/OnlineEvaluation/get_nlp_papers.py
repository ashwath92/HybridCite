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

# Get natural language processing contexts: fieldofstudyid=204321447: 17126 papers
query = """
     SELECT englishnlppapers.paperid, englishnlppapers.papertitle, citationcontexts.citationcontext, citationcontexts.paperreferenceid from
        (
            SELECT papertitle, nlppapers.paperid from 
            (
                SELECT papers.paperid, papers.originaltitle as papertitle FROM papers 
                 INNER JOIN 
                (SELECT paperid from paperfieldsofstudy WHERE fieldofstudyid=204321447) AS fieldsofstudy 
                 ON papers.paperid=fieldsofstudy.paperid where papers.publishedyear in (2018,2019)
            ) AS nlppapers
            INNER JOIN 
            (SELECT paperid FROM paperlanguages WHERE languagecode='en') AS languages 
            ON languages.paperid=nlppapers.paperid
        ) AS englishnlppapers INNER JOIN 
         
    (SELECT paperid, paperreferenceid, citationcontext
    FROM papercitationcontexts) AS citationcontexts 
    ON citationcontexts.paperid=englishnlppapers.paperid; """

cur.execute(query)
with open('/home/ashwath/Programs/OnlineEvaluation/AllNLPcontexts.tsv', 'w') as outfile:
    outfile.write('{}\t{}\t{}\t{}\n'.format('magid', 'title', 'context', 'groundtruth'))
    for row in cur:
        outfile.write('{}\t{}\t{}\t{}\n'.format(row['paperid'], row['papertitle'], row['citationcontext'], row['paperreferenceid']))