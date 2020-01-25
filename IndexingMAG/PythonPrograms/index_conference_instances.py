
import os
import pysolr
import requests
import csv

def insert_into_solr():
    """ Inserts records into an empty solr index which has already been created."""
    solr = pysolr.Solr('http://localhost:8983/solr/mag_conference_instances', always_commit=True)
    filepath = '/vol1/mag/data/2018-07-19/dumps/ConferenceInstances.txt'

    list_for_solr = []
    with open(filepath, "r") as file:
        csv_reader = csv.reader(file, delimiter='\t')
        field_names = ('author_id', 'rank', 'norm_name', 'display_name', 'affiliation_id',
                       'paper_count', 'citation_count', 'created_date')
        for conference_instance_id, rank, normalized_name, display_name, conference_series_id, location, official_url, \
                start_date, end_date, abstract_registration_date, submission_deadline_date, notification_due_date, \
                final_version_due_date, paper_count, citation_count, created_date in csv_reader:
        #for record in csv_reader:
            #print(len(record), record, type(record))
            solr_record = {}
            solr_record['conference_instance_id'] = conference_instance_id
            solr_record['rank'] = rank
            solr_record['normalized_name'] = normalized_name
            solr_record['display_name'] = display_name
            solr_record['conference_series_id'] = conference_series_id
            solr_record['location'] = location
            solr_record['official_url'] = official_url
            solr_record['start_date'] = start_date
            solr_record['end_date'] = end_date
            solr_record['abstract_registration_date'] = abstract_registration_date
            solr_record['submission_deadline_date'] = submission_deadline_date
            solr_record['notification_due_date'] = notification_due_date
            solr_record['final_version_due_date'] = final_version_due_date
            solr_record['paper_count'] = paper_count
            solr_record['citation_count'] = citation_count
            solr_record['created_date'] = created_date
            list_for_solr.append(solr_record)
        # Upload to Solr: there are only 15000-odd rows
        solr.add(list_for_solr)

if __name__ == '__main__':
    insert_into_solr()
