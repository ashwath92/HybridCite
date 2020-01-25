""" Prepapres a file from ACL data (as well as additional mag contexts) as per the hyperdoc2vec format.
All the citation markers in the output file are MAG Ids (for the ACL papers, the mapping from ACL to
MAG can be found in /home/ashwath/Programs/ACLAAn/SQLITEDB/acl_mag_mapping.sqlite3 -- table name: acl_mag).
Adjacent citations are not comma-separated, but instead just placed next to each other.
The input files have been preprocessed with citation markers replaced by the DBLP url or GC indicator.
The DBLP URLs which can be mapped are the only ones used to map to MAG (the title is obtained by scraping
DBLP) """

import os
import re
import sys
import pickle
import csv
import sqlite3
import pandas as pd
from time import sleep, time
import requests
from bs4 import BeautifulSoup
from gensim.parsing import preprocessing
from gensim.utils import to_unicode
import contractions
import psycopg2
import psycopg2.extras
from tqdm import tqdm

# POSTGRES connection obj and cursor
pconn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=shetland.informatik.uni-freiburg.de")
pcur = pconn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
query2 = "select paperid from papers where papertitle=%s;"

basepath = '/home/ashwath/Programs'
dbpath = os.path.join(basepath, 'ACLAAn', 'SQLITEDB', 'acl_mag_mapping.sqlite3')
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
replaced_citation_pattern = re.compile(r'(=-=)([0-9]+)(-=-)')
# IMPORTANT: I need a set of mag ids which are cited so that i can use it to add extra mag content.
allmagpaperids = set()
# mag acl mapping db connection
sconn = db_connect()
scur = sconn.cursor()

acl_filepath = '/home/ashwath/Programs/ACLAAn/acl-aan-text-sentences-refs'

def return_filepath_or_absent(acl_id):
    """ Forms a file path from the acl id, checks if it is present in the input directory,
    and returns either the full file path or None (NaN)"""
    filepath = '{}/{}.txt'.format(acl_filepath, acl_id)
    if os.path.exists(filepath):
        return filepath
    else:
        return None

# Training set is 1965-2005. 2006 is the test set)
# Training set: 9172 papers -- 9166 finally
# Test set: 1082 papers
trainingquery = """select acl_id, mag_id, published_year 
                     from acl_mag 
                     where  published_year<=2005 and published_year>=1965
                     group by mag_id;
                """
# Write test set
testsetquery = """select acl_id, mag_id, published_year
                     from acl_mag 
                     where published_year=2006
                     group by mag_id;

               """

testresdf = pd.read_sql_query(testsetquery, sconn)
testresdf['filename'] = testresdf['acl_id'].apply(return_filepath_or_absent)
testresdf = testresdf[~testresdf.filename.isnull()]
testresdf.to_csv('AdditionalOutputs/test_ids.tsv', index=False, sep='\t')

trainresdf = pd.read_sql_query(trainingquery, sconn)
trainresdf['filename'] = trainresdf['acl_id'].apply(return_filepath_or_absent)
# 6 rows get removed.
trainresdf = trainresdf[~trainresdf.filename.isnull()]
trainresdf.to_csv('AdditionalOutputs/training_ids.tsv', index=False, sep='\t')

# Get a Series of mag ids for which we have full text
mag_id_series = trainresdf['mag_id']
# IMP: There seems to be some problem with the data?? Multiple acl ids are mapped to the same mag id
# Doing select mag_id from acl_mag, and read_sql_query, then
# df[df.isin(df[df.duplicated()])] gives 69 records.
# Get a set of mag ids (mapped from acl of course) which have full text
inacl_papers_set = set(mag_id_series.tolist())

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

annotation_pattern = re.compile(r'<(GC|DBLP):([^>]+)>')
xml_p = re.compile(r'(a href=\")(http[s]?://dblp.org/[a-z0-9]+/xml/.*\.xml)')
headers = {'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}

# ACL citing, cited list based on mag ids 
acl_citing_cited_file = open('AdditionalOutputs/aclmag_references.tsv', 'w')
fieldnames = ['citing_mag_id', 'cited_mag_id']
writer = csv.DictWriter(acl_citing_cited_file, delimiter="\t", fieldnames=fieldnames)
writer.writeheader()

replaced_citation_pattern = re.compile(r'(=-=)([0-9]+)(-=-)')

def get_magid_from_annotation(matchobject):
    """ This takes a found dblp annotation, scrapes the website and gets the title, uses this
    to query mag, returns a mag id with doc id prefix and suffix"""
    cited_dblpurl = matchobject.group(2)
    try:
        res = requests.get(cited_dblpurl, headers=headers)
        # the xml is after 'export record'
        res = res.text[res.text.find('export record'):]
        xml_url_matchobj = xml_p.search(res)
        if xml_url_matchobj is None:
            return 'citation'
        xml_url = xml_url_matchobj.group(2)
        sleep(1) 
        print(xml_url)  
        xml_res = requests.get(xml_url, headers=headers)
        soup = BeautifulSoup(xml_res.content, 'lxml')
        title_bs4tag = soup.find('title')
        # If it can't find the title for whatever reason, remove the citation
        if title_bs4tag is None:
            return 'citation'
        title = title_bs4tag.string
        if title is None:
            return 'citation'
        title = preprocessing.strip_multiple_whitespaces(
            preprocessing.strip_punctuation(preprocessing.strip_tags(title.lower()))).strip()
        pcur.execute(query2, (title,))
        resultset = pcur.fetchone()
        if resultset is None:
            # If the uuid does not map to a mag id, replace with the word citation.
            #wordindex_magid_dict[i] = 'citation'
            print('not found')
            return 'citation'
        else:
            #print(resultset)
            fetched_magid = resultset['paperid']
            allmagpaperids.add(fetched_magid)
            return '{}{}{}'.format(docid_prefix, fetched_magid, docid_suffix)
    except requests.exceptions.MissingSchema:
        # for GC annotations
        return 'citation'

def write_refs_file(content, mag_id):
    """ writes into the refs file (citing paper id, cited paperid)"""
    for citationmarker in replaced_citation_pattern.finditer(content):
        # group(2) gets the magid from the match object
        fetched_mag_id = citationmarker.group(2)
        writer.writerow({'citing_mag_id': mag_id,'cited_mag_id': fetched_mag_id})

def read_file_addmagid(aclmag_list):
    """ Reads the ACL file, maps the dblp annotations to mag by scraping the
    dblp website, and returns the content. It returns None if the file is not
    present (metadata has additional papers which are not present in the
    preprocessed files"""
    acl_filepath = aclmag_list[0]
    mag_id = aclmag_list[1]
    allmagpaperids.add(mag_id)
    try:
        with open(acl_filepath, 'r') as aclfile:
            content = aclfile.read().replace('\n============\n', ' ').replace('\n', ' ')
    except FileNotFoundError:
        # Some files in the metadata are not present in the preprocessed files.
        # Safety check
        return None
    content = annotation_pattern.sub(get_magid_from_annotation, content)

    # Make sure to add the citing paper mag id as the first word in the line
    content = '{} {}\n'.format(mag_id, content)
    # Write to refs file:
    write_refs_file(content, mag_id)
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


def add_additional_papers(outfile):
    """ Add additional papers for which full text from ACL is not present. Care is taken that while
    adding references to THESE papers, these references should be in the set of papers stored
    in the allmagpaperids set (otherwise, there will be additional papers in the reference part
    of the concat contexts which are not in the files in the text.
    ALSO NOTE that allmagpaperids contains all papers which either cite or are cited so far
    inacl_papers_set contains the set of papers which are in acl (citing)
    A set difference (allmagpaperids - inacl_papers_set) gives the set of mag_ids for which we 
    get additional text"""

    additional_mag_ids = allmagpaperids - inacl_papers_set
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
                        writer.writerow({'citing_mag_id': paperid,'cited_mag_id': referenceid})
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
            if content.strip() != '':
                outfile.write(content)
                print("Written in file for {}".format(paperid))



if __name__ == '__main__':
    start = time()
    acl_filepath = '/home/ashwath/Programs/ACLAAn/acl-aan-text-sentences-refs'
    outfile = open('/home/ashwath/Programs/ACLAAn/acl_training_data.txt', 'w')
    trainresdf['acl_id'] = trainresdf['acl_id'].apply(lambda x: '{}/{}.txt'.format(acl_filepath, x))
    # aclmag_list is a list of lists
    aclmag_list = trainresdf[['filename', 'mag_id']].values.tolist()
    i=0
    for filename_and_magid in aclmag_list:
        print("file", i)
        i += 1
        content = read_file_addmagid(filename_and_magid)
        if content is None:
            # file can't be opened, just a safeguard
            continue
        outfile.write(content)

    with open('Pickles/inacl_papers_set.pickle', 'wb') as picc:
        pickle.dump(inacl_papers_set, picc)
    with open('Pickles/allmagpapers_en_magcontexts.pickle', 'wb') as picc2:
        pickle.dump(allmagpaperids, picc2)

    try:
        add_additional_papers(outfile)
    except Exception as e:
        print("additional papers not added. Exception:", e)
        outfile.close()
    # Clean up files and db connections.
    outfile.close()
    acl_citing_cited_file.close()
    sconn.close()
    pconn.close()
    print("Time taken:{}".format(time() - start))






