import pysolr
import requests
import os
import sqlite3
import psycopg2
import psycopg2.extras
from gensim.parsing import preprocessing

# GLOBALS
conn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=shetland.informatik.uni-freiburg.de")
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

solr = pysolr.Solr('http://localhost:8983/solr/arxiv_cs_metadata', always_commit=True)

def search_solr_parse_json(query, collection, search_field):
    """ Searches the arxiv_cs_metadata collection on arxiv_identifier (search_field)
    using the resp. arxiv id as the query, 
    parses the json result and returns it as a list of dictionaries where
    each dictionary corresponds to a record. 
    ARGUMENTS: query, string: each arxiv id
               collection: the Solr collection name (=arxiv_cs_metadata)
               search_field: the Solr field which is queried (=arxiv_identifier)
    RETURNS: docs, list of dicts: the documents (records) returned by Solr 
             AFTER getting the JSON response and parsing it."""
    solr_url = 'http://localhost:8983/solr/' + collection + '/select'
    url_params = {'q': query, 'rows': 1, 'df': search_field}
    solr_response = requests.get(solr_url, params=url_params)
    if solr_response.ok:
        data = solr_response.json()
        # Only one result, so index 0.
        docs = data['response']['docs']
        if docs == []:
            print(docs, query)
            return None, None
        doc = docs[0]
        title = doc.get('title').replace('\n', ' ')
        # Normalize the title
        title = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(title.lower()))
        published_year = doc.get('published_date')[:4]
        return title, published_year
    else:
        print("Invalid response returned from Solr")
        sys.exit(11)

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

def create_arxivcs_mag_table(conn):
    """ Function which takes a sqlite3 connection object and creates a table
    unpaywall if it doesn't exist. """ 
    cur = conn.cursor()
    mag_sql = """
    CREATE TABLE IF NOT EXISTS arxivcs_mag (
            arxiv_id text NOT NULL,
            mag_id text NOT NULL,
            PRIMARY KEY(mag_id, arxiv_id)
            )"""
    cur.execute(mag_sql)

def insert_into_arxivcs_mag(sconn, scur, arxiv_id, mag_id):
    """ Function which inserts paper_id, pdf_url and doi into unpaywall. It takes a sqlite3 conn object,
        as argument."""
    insert_arxivcs_mag_sql = """
    INSERT INTO arxivcs_mag(arxiv_id, mag_id)
    VALUES (?, ?) """
    scur.execute(insert_arxivcs_mag_sql, (arxiv_id, mag_id))

def select_from_arxivcs_mag(sconn, scur, paper_id):
    """ Queries the sqlite3 table unpaywall on paper_id, returns the pdf_url (can be None)"""
    # cur = conn.cursor()
    query = """
    SELECT pdf_url 
    FROM arxivcs_mag WHERE paper_id = '{}' 
    """.format(paper_id)
    scur.execute(query)
    # Only get one row (there will only be 1 row in the result). Only 1 field present.
    return scur.fetchone()[0] 

def main():
    """ Main function"""
    sconn = db_connect()
    scur = sconn.cursor()
    create_arxivcs_mag_table(sconn)
    with open('ArxivIDsCS/CS_arxiv_IDs.fixed', 'r') as arxividsfile, open('AdditionalOutputs/no_arxiv_mag_mapping.txt', 'w') as reject:
        for line in arxividsfile:
            arxivid = line.strip()
            title, publishedyear = search_solr_parse_json(arxivid, 'arxiv_cs_metadata', 'arxiv_identifier')
            if title is None:
                continue
            # Query postgres using title and published year first
            query1 = 'select paperid from papers where papertitle=%s and publishedyear=%s;'
            cur.execute(query1, (title, publishedyear))
            paperid = cur.fetchone()
            if paperid:
                paperid = paperid['paperid']
            query2 = "select paperid from papers where papertitle=%s;"
            if not paperid:
                # Try the query without the year
                cur.execute(query2, (title,))
                resultset = cur.fetchone()
                if not resultset:
                    # Skip this reference, not found in MAG
                    reject.write('{}\n'.format(arxivid))
                    continue
                paperid = resultset['paperid']
            # Create the sqlite table if it doesn't exist.
            
            insert_into_arxivcs_mag(sconn, scur, arxivid, paperid)

        try:
            sconn.commit()
        except:
            print("Something went wrong while committing, attempting to rollback!")
            sconn.rollback()
        scur.execute("select count(*) from arxivcs_mag")
        print("No. of records in db=", scur.fetchall())
        sconn.close()

if __name__ == '__main__':
    main()