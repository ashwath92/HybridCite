
import os
import pysolr
import requests
import csv

def insert_into_solr():
    """ Inserts records into an empty solr index which has already been created."""
    solr = pysolr.Solr('http://localhost:8983/solr/mag_journals', always_commit=True)
    filepath = '/vol1/mag/data/2018-07-19/dumps/Journals.txt'

    list_for_solr = []
    with open(filepath, "r") as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for journal_id, rank, normalized_name, display_name, issn, publisher, webpage, paper_count, citation_count, created_date in csv_reader:
            solr_record = {}
            solr_record['journal_id'] = journal_id
            solr_record['rank'] = rank
            solr_record['normalized_name'] = normalized_name
            solr_record['display_name'] = display_name
            solr_record['issn'] = issn
            solr_record['publisher'] = publisher
            solr_record['webpage'] = webpage
            solr_record['paper_count'] = paper_count
            solr_record['citation_count'] = citation_count
            solr_record['created_date'] = created_date
            list_for_solr.append(solr_record)
        # Upload to Solr: 48000-odd rows
        solr.add(list_for_solr)

if __name__ == '__main__':
    insert_into_solr()
