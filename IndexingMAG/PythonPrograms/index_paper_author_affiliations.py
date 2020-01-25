
import os
import pysolr
import requests
import csv

def insert_into_solr():
    """ Inserts records into an empty solr index which has already been created."""
    solr = pysolr.Solr('http://localhost:8983/solr/mag_paper_author_affiliations', always_commit=True)
    filepath = '/vol1/mag/data/2018-07-19/dumps/PaperAuthorAffiliations.txt'

    list_for_solr = []
    with open(filepath, "r") as file:
        csv_reader = csv.reader(file, delimiter='\t')
        field_names = ('author_id', 'rank', 'norm_name', 'display_name', 'affiliation_id',
                       'paper_count', 'citation_count', 'created_date')
        record_number = 0
        for paper_id, author_id, affiliation_id, author_sequence_number in csv_reader:
            record_number += 1
            solr_record = {}
            solr_record['paper_id'] = paper_id
            solr_record['author_id'] = author_id
            solr_record['affiliation_id'] = affiliation_id
            solr_record['author_sequence_number'] = author_sequence_number
            # Chunk and insert to Solr, chunks of 100000 (we only have small records, large chunks are okay)
            if record_number % 100000 == 0:
                list_for_solr.append(solr_record)
                solr.add(list_for_solr)
                list_for_solr = []
                print(record_number)
            else:
                list_for_solr.append(solr_record)
            
        solr.add(list_for_solr)

if __name__ == '__main__':
    insert_into_solr()
