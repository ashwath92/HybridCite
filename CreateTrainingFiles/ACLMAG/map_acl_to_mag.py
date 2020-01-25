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

def create_acl_mag_table(conn):
    """ Function which takes a sqlite3 connection object and creates a table
    unpaywall if it doesn't exist. """ 
    cur = conn.cursor()
    mag_sql = """
    CREATE TABLE IF NOT EXISTS acl_mag (
            acl_id text NOT NULL,
            mag_id text NOT NULL,
            published_year int,
            PRIMARY KEY(mag_id, acl_id)
            )"""
    cur.execute(mag_sql)

def insert_into_acl_mag(sconn, scur, acl_id, mag_id, published_year):
    """ Function which inserts paper_id, pdf_url and doi into unpaywall. It takes a sqlite3 conn object,
        as argument."""
    insert_acl_mag_sql = """
    INSERT INTO acl_mag(acl_id, mag_id, published_year)
    VALUES (?, ?, ?) """
    scur.execute(insert_acl_mag_sql, (acl_id, mag_id, published_year))

def select_from_acl_mag(sconn, scur, paper_id):
    """ Queries the sqlite3 table unpaywall on paper_id, returns the pdf_url (can be None)"""
    # cur = conn.cursor()
    query = """
    SELECT pdf_url 
    FROM acl_mag WHERE paper_id = '{}' 
    """.format(paper_id)
    scur.execute(query)
    # Only get one row (there will only be 1 row in the result). Only 1 field present.
    return scur.fetchone()[0] 

def main():
    """ Main function"""
    sconn = db_connect()
    scur = sconn.cursor()
    create_acl_mag_table(sconn)
    reject = open('AdditionalOutputs/no_acl_mag_mapping.txt', 'w')
    with open('Metadata/acl-metadata.txt', 'r', encoding='ISO-8859-1') as aclfile:
         
        content = aclfile.read()
        #'id = {D10-1001}\nauthor = {Rush, Alexander M.; Sontag, David; Collins, Michael John; Jaakkola, Tommi}
        #\ntitle = {On Dual Decomposition and Linear Programming Relaxations for Natural Language Processing}\n
        #venue = {EMNLP}\nyear = {2010}\n\nid = {D10-1002}\nauthor = {Huang, Zhongqiang; Harp'
    
    lines = content.split('\n\n')
    for line in lines:
        parts = line.split('\n')
        # 'id = {D10-1002}\nauthor = {Huang, Zhongqiang; Harper, Mary P.; Petrov, Slav}\ntitle = {Self-
        # Training with Products of Latent Variable Grammars}\nvenue = {EMNLP}\nyear = {2010}'
        acl_id = parts[0][parts[0].find('{')+1:parts[0].find('}')]
        title = parts[2][parts[2].find('{')+1:parts[2].find('}')]
        print(parts[4])
        publishedyear = int(parts[4][parts[4].find('{')+1:parts[4].find('}')])
        title = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(title.lower())).strip()
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
                reject.write('{}\n'.format(acl_id))
                continue
            paperid = resultset['paperid']
            
        insert_into_acl_mag(sconn, scur, acl_id, paperid, publishedyear)

    try:
        sconn.commit()
    except:
        print("Something went wrong while committing, attempting to rollback!")
        sconn.rollback()
    scur.execute("select count(*) from acl_mag")
    print("No. of records in db=", scur.fetchall())
    sconn.close()
    reject.close()

if __name__ == '__main__':
    main()