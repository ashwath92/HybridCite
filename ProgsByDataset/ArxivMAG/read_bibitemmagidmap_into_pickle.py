""" Write the results from the bibitemmagidmap database table at /vol2/unarXive/arxiv-txt-data/metadata.db
 into a dict with uuid keys and mag_id values. Lookup seems faster than Series, and much faster than df"""

import pandas as pd
import os
import pickle
import sqlite3

def db_connect(path, set_params=False):
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

# context connection: for getting the mag id of the CITED papers 
meta_db_path = '/vol2/unarXive/arxiv-txt-data/metadata.db'
cconn = db_connect(path=meta_db_path)
#ccur = cconn.cursor()

query = 'select uuid,mag_id from bibitemmagidmap'
df = pd.read_sql_query(query, cconn)
df = df.set_index('uuid')

# Convert to a a series for fast dict-like lookup with uuid as the key and mag_id as the value.

series = df.mag_id
# Convert to a dict to get even faster lookup with uuid keys and mag_id values
uuid_mag_dict = series.to_dict()

with open('Pickles/uuid_magid_dict.pickle', 'wb') as picc:
    pickle.dump(uuid_mag_dict, picc)
#series.to_pickle('Pickles/uuid_magid_series.pickle')


