import os
import sqlite3
import csv

basepath = '/home/ashwath/Files'
dbpath = os.path.join(basepath, 'Unpaywall', 'mag_unpaywall_mapping.sqlite3')

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

def create_mag_unpaywall_table(conn):
    """ Function which takes a sqlite3 connection object and creates a table
    unpaywall if it doesn't exist. """ 
    cur = conn.cursor()
    unpaywall_sql = """
    CREATE TABLE IF NOT EXISTS mag_unpaywall (
            paper_id text NOT NULL,
            doi text NOT NULL,
            pdf_url text NOT NULL,
            PRIMARY KEY(paper_id)
            )"""
    cur.execute(unpaywall_sql)

def insert_into_mag_unpaywall(conn):
    """ Function which inserts paper_id, pdf_url and doi into unpaywall. It takes a sqlite3 conn object,
        as argument."""
    cur = conn.cursor()
    insert_mag_unpaywall_sql = """
    INSERT INTO mag_unpaywall(paper_id, doi, pdf_url)
    VALUES (?, ?, ?) """
    filepath = '/home/ashwath/Files/Unpaywall/mag_unpaywall_mapping.tsv'
    with open(filepath, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter='\t')
        # IMPORTANT: the file has a header, this needs to be skipped
        next(csv_reader, None)   
        cur.executemany(insert_mag_unpaywall_sql, csv_reader)
        #for paper_id, doi, pdf_url in csv_reader:
        #    cur.execute(insert_mag_unpaywall_sql, (paper_id, doi, pdf_url))

def select_from_mag_unpaywall(conn, cur, paper_id):
    """ Queries the sqlite3 table unpaywall on paper_id, returns the pdf_url (can be None)"""
    # cur = conn.cursor()
    query = """
    SELECT pdf_url 
    FROM unpaywall WHERE paper_id = '{}' 
    """.format(paper_id)
    cur.execute(query)
    # Only get one row (there will only be 1 row in the result). Only 1 field present.
    return cur.fetchone()[0] 

if __name__ == '__main__':
    conn = db_connect()
    # UNCOMMENT THE CREATE AND INSERT INTO (and commit) STATEMENTS TO do a fresh create/insert.
    cur = conn.cursor()
    cur.execute('drop table mag_unpaywall')
    create_mag_unpaywall_table(conn)
    insert_into_mag_unpaywall(conn)
    try:
        conn.commit()
    except:
        print("Something went wrong while committing, attempting to rollback!")
        conn.rollback()
    #cur = conn.cursor()
    cur.execute("select count(*) from mag_unpaywall")
    print("No. of records in db=", cur.fetchall())
    conn.close()
