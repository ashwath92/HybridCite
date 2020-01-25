"""Evaluation metrics:  
Recall@k    k=1,..,10
MAP@k   k=1,..,10
nDCG@k  k=1,..,10
MRR@k   k=1,..,10"""
# /home/saiert/bk/master_thesis/code/recommender_claim/context_exports_jama/items_acl-arc_DBLPonly.csv
import argparse
from time import sleep
from bs4 import BeautifulSoup
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from gensim.parsing import preprocessing
import pickle
import re
import csv
import os
import pandas as pd
from tqdm import tqdm

# allmagpaperids has all the mag paper ids in the training set. The dblp to mag mapping has unfortunately not
# been stored.
with open('Pickles/trainingmagids.pickle', 'rb') as picc2:
    allmagpaperids = pickle.load(picc2)

# Get the list of test ids
# 1082 test set ids
testsetids = set(pd.read_csv('/home/ashwath/Programs/ACLAAn/AdditionalOutputs/test_ids.tsv', sep='\t')['acl_id'].tolist())
# dblp to mag mapping
dblp_mag_map_df = pd.read_csv('AdditionalOutputs/TESTFILE_dblp_to_mag.tsv', sep='\t', names=['dblp_url','mag_id'])

# POSTGRES connection obj and cursor
pconn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=shetland.informatik.uni-freiburg.de")
pcur = pconn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
query2 = "select paperid from papers where papertitle=%s;"


# Send a post request for eahc dblp url, get the text and parse it, find the xml obj, get the title, map to
# mag. If it can be mapped, keep the context, if not discard it (primary citation id)
xml_p = re.compile(r'(a href=\")(http[s]?://dblp.org/[a-z0-9]+/xml/.*\.xml)')

# Headers for the http request
headers = {'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}

fieldnames = ['dblp_url', 'mag_id']

#mappingfile = open('AdditionalOutputs/TESTFILE_dblp_to_mag.tsv', 'w')
#writer = csv.DictWriter(mappingfile, delimiter="\t", fieldnames=fieldnames)


def map_dblp_to_mag_requests(dblp_url):
    """ Takes a dblp url, gets the title by scraping the website and getting the relevant xml file, and using
    that to get the title. This title is used to map to MAG."""
    try:
        res = requests.get(dblp_url, headers=headers)
        # the xml is after 'export record'
        res = res.text[res.text.find('export record'):]
        xml_url_matchobj = xml_p.search(res)
        if xml_url_matchobj is None:
            return None
        xml_url = xml_url_matchobj.group(2)
        sleep(1) 
        print(xml_url)  
        xml_res = requests.get(xml_url, headers=headers)
        soup = BeautifulSoup(xml_res.content, 'lxml')
        title_bs4tag = soup.find('title')
        # If it can't find the title for whatever reason, remove the citation
        if title_bs4tag is None:
            return None
        title = title_bs4tag.string
        if title is None:
            return None
        title = preprocessing.strip_multiple_whitespaces(
            preprocessing.strip_punctuation(preprocessing.strip_tags(title.lower()))).strip()
        pcur.execute(query2, (title,))
        resultset = pcur.fetchone()
        if resultset is None:
            # If the uuid does not map to a mag id, replace with the word citation.
            #wordindex_magid_dict[i] = 'citation'
            print('not found')
            return None
        else:
            #print(resultset)
            fetched_magid = resultset['paperid']
            #writer.writerow({'dblp_url': dblp_url, 'mag_id': fetched_magid})
            #allmagpaperids.add(fetched_magid)
            return fetched_magid
    except requests.exceptions.MissingSchema:
        # for GC annotations
        return None

def map_dblp_to_mag(dblp_url):
    """ Function for subsequent runs. the xml files have been fetched and a mapping file
    has already been created"""
    try:
        return repr(dblp_mag_map_df.loc[dblp_mag_map_df.dblp_url==dblp_url]['mag_id'].values[0])
    except IndexError:
        # when there is no match
        return None

with open('/home/ashwath/Files/items_acl-arc_DBLPonly.csv', 'r') as infile,\
     open('/home/ashwath/Programs/ACLAAn/acl_testset_contexts.tsv', 'w') as outfile:
     for line in tqdm(infile):
        parts = line.split('\u241E')
        acl_id = parts[2]
        # Skip if the acl id is not in the test set
        if acl_id not in testsetids:
            continue
        citeddblpurl = parts[0]
        citedmagid = map_dblp_to_mag(citeddblpurl)
        if citedmagid is None or citedmagid not in allmagpaperids:
            # If the main cited dblp URL cannot be mapped, ignore the context. 
            # Also, if it hasn't been seen in the training data, ignore the context.
            continue
        citedmagids = [citedmagid]
        if parts[1] != '':
            print("Additional ids:,", parts[1])
        # Additional cited mag ids are separtated by a field separator, 
        additionaldblpurls = parts[1].split('\u241F') 
        if additionaldblpurls != ['']:
            additionalmagids = [map_dblp_to_mag(additionaldblpurl) for additionaldblpurl in additionaldblpurls]
            # Chuck out dblp urls which couldn't be mapped to mag ids
            # Also, discard mag ids which are not in the training data
            additionalmagids = [additionalmagid for additionalmagid in additionalmagids 
                                                if additionalmagid is not None and additionalmagid in allmagpaperids]
            
            if additionalmagids != []:
                citedmagids.extend(additionalmagids)
        citedmagids_string = ','.join(citedmagids)
        context = preprocessing.strip_multiple_whitespaces(parts[3].replace('MAINCIT', ' ').replace('CIT', '').replace(' ,', '')).replace('- ', '')
        outfile.write('{}\t{}\t{}\n'.format(citedmagids_string, acl_id, context))        

#mappingfile.close()
'''query = """
     SELECT englishcspapers.paperid, englishcspapers.papertitle, citationcontexts.citationcontext, 
     citationcontexts.paperreferenceid from
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
    FROM papercitationcontexts GROUP BY paperid, citationcontext) AS citationcontexts 
    ON citationcontexts.paperid=englishcspapers.paperid; """

#testresdf = pd.read_sql_query(query, sconn)
 
Same contexts with different spacing, the only possible way to combine them would be to remove spaces and punctuations.
  473070 | a simple scheme for contour detection | and Georgescu, 2001), anisotropic diffusion (Perona and Malik, 1990; Black et al., 1998), complementary analysis of boundaries and regions ( Ma and Manjunath, 2000 ) and edge density information (Dubuc and Zucker, 2001).                                                                                               | 2145011027
  473070 | a simple scheme for contour detection | and Georgescu, 2001), anisotropic diffusion (Perona and Malik, 1990; Black et al., 1998), complementary analysis of boundaries and regions (Ma and Manjunath, 2000) and edge density information ( Dubuc and Zucker, 2001 ).                                                                                               | 1593431850
  473070 | a simple scheme for contour detection | and Georgescu, 2001), anisotropic diffusion (Perona and Malik, 1990; Black et al., 1998), complementary analysis of boundaries and regions (Ma and Manjunath, 2000) and edge density information (Dubuc and Zucker, 2001).     



parser = argparse.Ar


'''