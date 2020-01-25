import os
import argparse
import sqlite3
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests.utils import requote_uri
import csv
import urllib.parse
from time import sleep

basepath = '/vol3/mag/Unpaywall'
dbpath = os.path.join(basepath, 'mag_unpaywall_mapping.sqlite3')

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

def select_from_mag_unpaywall(conn, cur, paper_id):
    """ Queries the sqlite3 table unpaywall on paper_id, returns the pdf_url (can be None)"""
    #cur = conn.cursor()
    query = """
    SELECT pdf_url 
    FROM mag_unpaywall WHERE paper_id = '{}' 
    """.format(paper_id)
    cur.execute(query)
    # Only get one row (there will only be 1 row in the result). Only 1 field present.
    return cur.fetchone()


def read_and_query(filepath, field, notfound_writer):
    """ Read the input file and download the pdfs based on a SQLITE3 query.
    """
    filepath = filepath + field + '.txt'
    output_filepath = os.path.join(basepath, 'Fulltext', 'pdfs', field)
    # Create the ouptut path if it doesn't exist
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    conn = db_connect()
    cur = conn.cursor()
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            paper_id = line.strip()
            pdf_url = select_from_mag_unpaywall(conn, cur, paper_id)
            if pdf_url is not None:
                download_pdf(pdf_url[0], paper_id, output_filepath, notfound_writer)
                if i%5 == 0:
                    # Sleep after every 5 papers
                    sleep(1)

def download_pdf(pdf_url, paper_id, output_filepath, notfound_writer):
    """ Downloads the pdf specified in pdf url and sets its name to mag paper_id."""
    try:
        # VERY IMPORTANT: Add User Agent in the Content header. When a page redirects, it produces a ConnectionError.
        # Adding the user agent prevents this and gets the response from the redirected page (allow_redirects=True by default)
        # https://stackoverflow.com/questions/36749376/python-issues-with-httplib-requests-https-seems-to-cause-a-redirect-then-bad
        user_agent_header = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36"}
        # URL-Encode PDF URL: spaces (instead of +) etc might be present or may have been introduced while inserting into SQLITE3. T
        #pdf_url = urllib.parse.quote_plus(pdf_url.strip())
        pdf_url = requote_uri(pdf_url.strip())
        # https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        url_response = requests.get(pdf_url, stream=True, timeout=45, headers=user_agent_header, verify=False)
        # print(url_response.url) -> redirected url
        filename = '{output_filepath}/{paper_id}.pdf'.format(output_filepath=output_filepath, paper_id=paper_id)
        with open(filename, 'wb') as pdf_file:
            # As the file may be large, write to it in chunks of 2000 bytes
            for chunk in url_response.iter_content(chunk_size=2000):
                pdf_file.write(chunk)

    except requests.exceptions.ConnectionError as ce:
        url_response = "No response"
        notfound_writer.writerow({'paper_id': paper_id, 'pdf_url': pdf_url, 'exception': ce})
        print('{} ({}) not found'.format(paper_id, pdf_url))
        # Go to the next line in the input file if the pdf is not found
        return

    except requests.exceptions.Timeout as te:
        url_response = "Timeout"
        notfound_writer.writerow({'paper_id': paper_id, 'pdf_url': pdf_url, 'exception': te})
        print('{} ({}) not found'.format(paper_id, pdf_url))
        # Go to the next line in the input file if the pdf is not found
        return

    except requests.exceptions.RequestException as e:
        # Catch all other exceptions
        url_response = "No response"
        notfound_writer.writerow({'paper_id': paper_id, 'pdf_url': pdf_url, 'exception': e})
        print('{} ({}) not found'.format(paper_id, pdf_url))
        # Go to the next line in the input file if the pdf is not found
        return

    except requests.packages.urllib3.exceptions.LocationParseError as e:
        # Catch all other exceptions
        url_response = "Location error"
        notfound_writer.writerow({'paper_id': paper_id, 'pdf_url': pdf_url, 'exception': e})
        print('{} ({}) not found'.format(paper_id, pdf_url))
        # Go to the next line in the input file if the pdf is not found
        return

    except Exception as e:
        # catch everything except KeyboardInterrupt and SystemExit. Could be trouble, but as so many
        # diff exceptions are being raised each time, I'm going with it.
        url_response = "Some error"
        notfound_writer.writerow({'paper_id': paper_id, 'pdf_url': pdf_url, 'exception': e})
        print('{} ({}) not found'.format(paper_id, pdf_url))
        # Go to the next line in the input file if the pdf is not found
        return        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('field', help="Enter field of study (all small letters, no spaces between words)")
    args = parser.parse_args()
    filepath = '/home/ashwath/Files/PapersByFieldsOfStudy/'
    field = args.field.strip()
    # Create a file which contains the pdfs which could not be downloaded. 
    fieldnames = ['paper_id', 'pdf_url', 'exception']
    notfound_path = os.path.join(basepath, 'Fulltext', 'NotFound')
    notfound_tsv = open('/vol3/mag/Unpaywall/Fulltext/NotFound/pdf_not_found_{}.txt'.format(field), 'w')
    notfound_writer = csv.DictWriter(notfound_tsv, delimiter='\t', fieldnames=fieldnames)
    #notfound_writer.writeheader()
    # Call read and query, which in turn calls the download function
    read_and_query(filepath, field, notfound_writer)
    notfound_writer.close()

if __name__ == '__main__':
    main()