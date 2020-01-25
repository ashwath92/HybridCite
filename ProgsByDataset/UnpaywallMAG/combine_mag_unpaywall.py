
import os
import csv
import sqlite3
from tqdm import tqdm
import parse_unpaywall


def create_final_file():
    """ Creates a file from Papers.txt and the unpaywall data which is stored in sqlite3. The pdf URL is fetched
    from the database and added to the final file."""
    # Create a connection object for the sqlite database
    conn = parse_unpaywall.db_connect()
    cur = conn.cursor()
    # Create a CSV dict writer
    outputcsv = open('/home/ashwath/Files/Unpaywall/mag_unpaywall_mapping.tsv', 'w')
    fieldnames = ['paper_id', 'doi', 'pdf_url']
    writer = csv.DictWriter(outputcsv, delimiter='\t', fieldnames=fieldnames)
    writer.writeheader()
    record_no = 0
    #with open('/vol1/mag/data/2018-07-19/dumps/Papers.txt', "r") as file:
    with open('/home/ashwath/Files/Unpaywall/Papers.txt', "r") as file: #THE NEW FILE HAS 22 FIELDS!!!!
        csv_reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
        #for paper_id, rank, doi, doc_type, normalized_title, original_title, book_title, publication_year, publication_date, \
        #        publisher, journal_id, conference_series_id, conference_instance_id, volume, issue, first_page, last_page, \
        #        reference_count, citation_count, estimated_citation_count, created_date in csv_reader:
        #    print(paper_id, rank, doi, doc_type, normalized_title, original_title, book_title, publication_year, publication_date, \
        #        publisher, journal_id, conference_series_id, conference_instance_id, len(volume), type(issue), len(first_page), len(last_page), \
        #        reference_count, citation_count, estimated_citation_count, created_date)
        for record in tqdm(csv_reader):
            #print(len(record))
            if len(record) < 22:
                # Too few fields 
                errorrecord = 'BAD RECORD: TOO FEW FIELDS ({} instead of 22)!! paper_id = {}\n'.format(len(record), record[0])
                continue

            record_no += 1
            paper_id, rank, doi, doc_type, normalized_title, original_title, book_title, publication_year, publication_date, \
            publisher, journal_id, conference_series_id, conference_instance_id, volume, issue, first_page, last_page, \
            reference_count, citation_count, estimated_citation_count, original_venue, created_date = record

            # Get the pdf url (it's like an inner join based on doi as the key)
            if doi is not None:
                pdf_url = parse_unpaywall.select_from_unpaywall(conn, cur, doi.strip())
                # Check if a result is returned
                if pdf_url is not None:
                    # Also check if the value of the pdf_url in the database (and therefore in the unpaywall jsonl file) was None (null in file)
                    if pdf_url[0] is not None:
                        writer.writerow({'paper_id': paper_id, 'doi': doi, 'pdf_url': pdf_url[0]})

            
    outputcsv.close()
    conn.close()

if __name__ == '__main__':
    create_final_file()
