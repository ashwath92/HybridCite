import jsonlines
import os
import sqlite3

basepath = '/home/ashwath/Files'
dbpath = os.path.join(basepath, 'Unpaywall', 'unpaywall_doi_pdf.sqlite3')

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

def create_unpaywall_table(conn):
    """ Function which takes a sqlite3 connection object and creates a table
    unpaywall if it doesn't exist. """ 
    cur = conn.cursor()
    unpaywall_sql = """
    CREATE TABLE IF NOT EXISTS unpaywall (
    doi text NOT NULL,
            pdf_url text,
            PRIMARY KEY(doi)
            )"""
    cur.execute(unpaywall_sql)

def insert_into_unpaywall(conn):
    """ Function which inserts pdf_url and doi into unpaywall. It takes a sqlite3 conn object,
        filename as argument."""
        
    cur = conn.cursor()
    insert_unpaywall_sql = """
    INSERT INTO unpaywall(doi, pdf_url)
    VALUES (?, ?) """
    with jsonlines.open('/home/ashwath/Files/Unpaywall/filtered_unpaywall_2018-09-24.jsonl') as reader:
        for line in reader:
            # Initialize pdf_url to None, as if none of the if conditions are satisfied, we want to store None
            if line['best_oa_location']['url_for_pdf'] != 'null':
                pdf_url = line['best_oa_location']['url_for_pdf']
            # If best location url is null, cycle through all the locations, and if one of them is not null, insert them
            # into pdf_url instead. If all are null, pdf_url = None (it is initialized as None above)
            else:
                for location in line['oa_locations']:
                    # line['oa_locations'] is a list of dicts
                    if location['url_for_pdf'] != 'null':
                        pdf_url = location['url_for_pdf']
                        # We have a url, no need of any more iterations. Break now.
                        break
            
            #print(line['best_oa_location']['url_for_pdf'], line['doi'])
            #print(reader.read()['best_oa_location'])
            cur.execute(insert_unpaywall_sql, (line['doi'], pdf_url))
        
    
def select_from_unpaywall(conn, cur, doi):
    """ Queries the sqlite3 table unpaywall on doi, returns the pdf_url (can be None)"""
    # cur = conn.cursor()
    query = """
    SELECT pdf_url 
    FROM unpaywall WHERE doi = '{}' 
    """.format(doi)
    cur.execute(query)
    # Only get one row (there will only be 1 row in the result). 1 field only present.
    return cur.fetchone()  

def count_null_urls(conn):
    """ Queries the sqlite3 table unpaywall and finds the number of Nones in pdf_url"""
    cur = conn.cursor()
    query = """
    SELECT count(*) 
    FROM unpaywall 
    WHERE pdf_url IS NOT NULL """
    cur.execute(query)
    return cur.fetchone()[0]  

if __name__ == '__main__':
    conn = db_connect()

    # UNCOMMENT THE CREATE AND INSERT INTO (and commit) STATEMENTS TO do a fresh create/insert.
    #create_unpaywall_table(conn)
    #print(count_null_urls(conn))
    #print(select_from_unpaywall(conn, '10.1210/endo.141.6.7513'))
    #insert_into_unpaywall(conn)
    #try:
    #    conn.commit()
    #except:
    #    print("Something went wrong while committing, attempting to rollback!")
    #    conn.rollback()
    cur = conn.cursor()
    cur.execute("select count(*) from unpaywall")
    print(cur.fetchall())
    conn.close()
#    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
#    cur.execute("SELECT sql FROM sqlite_master WHERE type='table'")


############################################################################################
""" Explanation of input data
>> with jsonlines.open('filtered_unpaywall_2018-09-24.jsonl') as reader:
       x= reader.read() -> Get one record

Let's assume we get the first sample document below and store it in x. Let's store the second sample document in y.
A jsonline will have one best location (open access source: best_oa_location), and multiple other locations (oa_locations).
x['best_oa_location'] returns a dictionary, while x['oa_locations'] returns a list of dictionaries (only one present in this case for x). 
The best location is one of these dictionaries, marked with is_best = True. 
For sample document y, y['oa_locations'] actually returns a list of 2 dicts, the first is the best one (same as y['best_oa_location']) and 
the second is not the best (here, the first is the published version and the second is the submitted version).

What I'm doing in the code is getting the best_oa_locations url_for_pdf (primary url). But if the primary url is 'null' in the unpaywall json file,
I'm going through the oa_locations (all of them) to check if they have a url which is not 'null'. If yes, I add that to the sqlite3 table. 

NOTE: FOR NOW, I AM ONLY ADDING 1 URL IN THE SQLITE3 TABLE (the best url or the first of the other urls which is not 'null') 
IF NEEDED IN THE FUTURE, I can add BEST URL IN THE SQLITE TABLE as best_pdf_url, add any others as 'other_pdf_urls'

>>> x.keys()
dict_keys(['doi_url', 'is_oa', 'best_oa_location', 'journal_issns', 'publisher', 'journal_name', 'year', 'doi', 'data_standard', 'oa_locations', 'title', 'journal_is_in_doaj', 'z_authors', 'published_date', 'updated', 'genre', 'journal_is_oa'])

>>> x['best_oa_location']
{'version': 'publishedVersion', 'evidence': 'open (via free pdf)', 'url_for_pdf': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E1FDAA965C5B67C12274B2B3408AE206/S2059163217000019a.pdf/div-class-title-qing-military-institutions-and-their-effects-on-government-economy-and-society-1640-1800-div.pdf', 'updated': '2018-02-10T22:38:57.165805', 'url_for_landing_page': 'https://doi.org/10.1017/jch.2017.1', 'is_best': True, 'license': None, 'host_type': 'publisher', 'url': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E1FDAA965C5B67C12274B2B3408AE206/S2059163217000019a.pdf/div-class-title-qing-military-institutions-and-their-effects-on-government-economy-and-society-1640-1800-div.pdf', 'pmh_id': None}

>>> x['oa_locations']
[{'version': 'publishedVersion', 'evidence': 'open (via free pdf)', 'url_for_pdf': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E1FDAA965C5B67C12274B2B3408AE206/S2059163217000019a.pdf/div-class-title-qing-military-institutions-and-their-effects-on-government-economy-and-society-1640-1800-div.pdf', 'updated': '2018-02-10T22:38:57.165805', 'url_for_landing_page': 'https://doi.org/10.1017/jch.2017.1', 'is_best': True, 'license': None, 'host_type': 'publisher', 'url': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E1FDAA965C5B67C12274B2B3408AE206/S2059163217000019a.pdf/div-class-title-qing-military-institutions-and-their-effects-on-government-economy-and-society-1640-1800-div.pdf', 'pmh_id': None}]

>>> x['oa_locations'][0] == x['best_oa_location']
True

>>> y['best_oa_location']
{'version': 'publishedVersion', 'pmh_id': None, 'url_for_landing_page': 'https://doi.org/10.1017/s0515036100014124', 'evidence': 'open (via free pdf)', 'url_for_pdf': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F818D814C6A0A64BCB1CD31F7853835/S0515036100014124a.pdf/div-class-title-tail-conditional-expectations-for-exponential-dispersion-models-div.pdf', 'updated': '2018-06-09T07:01:36.778627', 'is_best': True, 'host_type': 'publisher', 'url': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F818D814C6A0A64BCB1CD31F7853835/S0515036100014124a.pdf/div-class-title-tail-conditional-expectations-for-exponential-dispersion-models-div.pdf', 'license': None}

>>> y['oa_locations']
[{'version': 'publishedVersion', 'evidence': 'open (via free pdf)', 'url_for_pdf': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F818D814C6A0A64BCB1CD31F7853835/S0515036100014124a.pdf/div-class-title-tail-conditional-expectations-for-exponential-dispersion-models-div.pdf', 'updated': '2018-06-09T07:01:36.778627', 'url_for_landing_page': 'https://doi.org/10.1017/s0515036100014124', 'is_best': True, 'license': None, 'host_type': 'publisher', 'url': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F818D814C6A0A64BCB1CD31F7853835/S0515036100014124a.pdf/div-class-title-tail-conditional-expectations-for-exponential-dispersion-models-div.pdf', 'pmh_id': None}, {'version': 'submittedVersion', 'evidence': 'oa repository (via OAI-PMH title and first author match)', 'url_for_pdf': 'http://wwwdocs.fce.unsw.edu.au/actuarial/research/papers/2003/TCE_Landsman_Valdez.pdf', 'updated': '2018-01-16T13:47:20.533609', 'url_for_landing_page': 'http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.494.6957', 'is_best': False, 'license': None, 'host_type': 'repository', 'url': 'http://wwwdocs.fce.unsw.edu.au/actuarial/research/papers/2003/TCE_Landsman_Valdez.pdf', 'pmh_id': 'oai:CiteSeerX.psu:10.1.1.494.6957'}]


Sample document (x):
{
    'doi_url': 'https://doi.org/10.1017/jch.2017.1',
    'is_oa': True,
    'best_oa_location': {
        'version': 'publishedVersion',
        'host_type': 'publisher',
        'url_for_landing_page': 'https://doi.org/10.1017/jch.2017.1',
        'evidence': 'open (via free pdf)',
        'url_for_pdf': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E1FDAA965C5B67C12274B2B3408AE206/S2059163217000019a.pdf/div-class-title-qing-military-institutions-and-their-effects-on-government-economy-and-society-1640-1800-div.pdf',
        'updated': '2018-02-10T22:38:57.165805',
        'is_best': True,
        'pmh_id': None,
        'url': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E1FDAA965C5B67C12274B2B3408AE206/S2059163217000019a.pdf/div-class-title-qing-military-institutions-and-their-effects-on-government-economy-and-society-1640-1800-div.pdf',
        'license': None
    },
    'journal_issns': '2059-1632,2059-1640',
    'publisher': 'Cambridge University Press (CUP)',
    'journal_name': 'Journal of Chinese History',
    'year': 2017,
    'doi': '10.1017/jch.2017.1',
    'data_standard': 2,
    'oa_locations': [{
        'version': 'publishedVersion',
        'host_type': 'publisher',
        'url_for_landing_page': 'https://doi.org/10.1017/jch.2017.1',
        'evidence': 'open (via free pdf)',
        'url_for_pdf': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E1FDAA965C5B67C12274B2B3408AE206/S2059163217000019a.pdf/div-class-title-qing-military-institutions-and-their-effects-on-government-economy-and-society-1640-1800-div.pdf',
        'updated': '2018-02-10T22:38:57.165805',
        'is_best': True,
        'pmh_id': None,
        'url': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E1FDAA965C5B67C12274B2B3408AE206/S2059163217000019a.pdf/div-class-title-qing-military-institutions-and-their-effects-on-government-economy-and-society-1640-1800-div.pdf',
        'license': None
    }],
    'title': 'QING MILITARY INSTITUTIONS AND THEIR EFFECTS ON GOVERNMENT, ECONOMY, AND SOCIETY, 1640–1800',
    'journal_is_in_doaj': False,
    'z_authors': [{
        'given': 'Yingcong',
        'family': 'Dai'
    }],
    'published_date': '2017-07-01',
    'updated': '2018-06-18T04:50:43.613017',
    'genre': 'journal-article',
    'journal_is_oa': False
}

Sample document 2 (y)
{
    'doi_url': 'https://doi.org/10.1017/s0515036100014124',
    'is_oa': True,
    'best_oa_location': {
        'version': 'publishedVersion',
        'host_type': 'publisher',
        'url_for_landing_page': 'https://doi.org/10.1017/s0515036100014124',
        'evidence': 'open (via free pdf)',
        'url_for_pdf': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F818D814C6A0A64BCB1CD31F7853835/S0515036100014124a.pdf/div-class-title-tail-conditional-expectations-for-exponential-dispersion-models-div.pdf',
        'updated': '2018-06-09T07:01:36.778627',
        'is_best': True,
        'pmh_id': None,
        'url': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F818D814C6A0A64BCB1CD31F7853835/S0515036100014124a.pdf/div-class-title-tail-conditional-expectations-for-exponential-dispersion-models-div.pdf',
        'license': None
    },
    'journal_issns': '0515-0361,1783-1350',
    'publisher': 'Cambridge University Press (CUP)',
    'journal_name': 'ASTIN Bulletin',
    'year': 2005,
    'doi': '10.1017/s0515036100014124',
    'data_standard': 2,
    'oa_locations': [{
        'version': 'publishedVersion',
        'host_type': 'publisher',
        'url_for_landing_page': 'https://doi.org/10.1017/s0515036100014124',
        'evidence': 'open (via free pdf)',
        'url_for_pdf': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F818D814C6A0A64BCB1CD31F7853835/S0515036100014124a.pdf/div-class-title-tail-conditional-expectations-for-exponential-dispersion-models-div.pdf',
        'updated': '2018-06-09T07:01:36.778627',
        'is_best': True,
        'pmh_id': None,
        'url': 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F818D814C6A0A64BCB1CD31F7853835/S0515036100014124a.pdf/div-class-title-tail-conditional-expectations-for-exponential-dispersion-models-div.pdf',
        'license': None
    }, {
        'version': 'submittedVersion',
        'host_type': 'repository',
        'url_for_landing_page': 'http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.494.6957',
        'evidence': 'oa repository (via OAI-PMH title and first author match)',
        'url_for_pdf': 'http://wwwdocs.fce.unsw.edu.au/actuarial/research/papers/2003/TCE_Landsman_Valdez.pdf',
        'updated': '2018-01-16T13:47:20.533609',
        'is_best': False,
        'pmh_id': 'oai:CiteSeerX.psu:10.1.1.494.6957',
        'url': 'http://wwwdocs.fce.unsw.edu.au/actuarial/research/papers/2003/TCE_Landsman_Valdez.pdf',
        'license': None
    }],
    'title': 'Tail Conditional Expectations for Exponential Dispersion Models',
    'journal_is_in_doaj': False,
    'z_authors': [{
        'given': 'Zinoviy',
        'family': 'Landsman'
    }, {
        'given': 'Emiliano A.',
        'family': 'Valdez'
    }],
    'published_date': '2005-05-01',
    'updated': '2018-06-19T09:54:13.063368',
    'genre': 'journal-article',
    'journal_is_oa': False
}

"""