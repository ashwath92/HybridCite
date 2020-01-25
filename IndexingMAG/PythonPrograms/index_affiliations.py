
import os
import pysolr
import requests
import csv

def insert_into_solr():
    """ Inserts records into an empty solr index which has already been created."""
    solr = pysolr.Solr('http://localhost:8983/solr/mag_affiliations', always_commit=True)
    filepath = '/vol1/mag/data/2018-07-19/dumps/Affiliations.txt'

    list_for_solr = []
    with open(filepath, "r") as file:
        csv_reader = csv.reader(file, delimiter='\t')
        field_names = ('affiliation_id', 'rank', 'norm_name', 'display_name', 'grid_id', 'homepage',
                       'wikipedia_url', 'paper_count', 'citation_count', 'created_date')
        for affiliation_id, rank, norm_name, display_name, grid_id, homepage, wikipedia_url, paper_count, citation_count, created_date in csv_reader:
            solr_record = {}
            solr_record['affiliation_id'] = affiliation_id
            solr_record['rank'] = rank
            solr_record['normalized_name'] = norm_name
            solr_record['display_name'] = display_name
            solr_record['grid_id'] = grid_id
            solr_record['homepage'] = homepage
            solr_record['wikipedia_url'] = wikipedia_url
            solr_record['paper_count'] = paper_count
            solr_record['citation_count'] = citation_count
            solr_record['created_date'] = created_date
            list_for_solr.append(solr_record)
        # Upload to Solr: there are only 25374 rows
        solr.add(list_for_solr)

if __name__ == '__main__':
    insert_into_solr()