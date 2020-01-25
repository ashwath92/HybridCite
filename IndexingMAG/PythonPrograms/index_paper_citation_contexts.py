
import os
import pysolr
import requests
import csv
from glob import glob
import concurrent.futures
from time import time
solr = pysolr.Solr('http://localhost:8983/solr/mag_paper_citation_contexts', always_commit=True)
    
def insert_into_solr(filepath):
    """ Inserts records into an empty solr index which has already been created."""
    #filepath = '/vol1/mag/data/2018-07-19/dumps/PaperCitationContexts.txt'
    record_number = 0
    list_for_solr = []
    with open(filepath, "r") as file:
        # THERE ARE NULL BYTES WHICH MAKE CSV THROW AN ERROR. Replace them 
        csv_reader = csv.reader((line.replace('\0', '') for line in file), delimiter='\t', quoting=csv.QUOTE_NONE)
        for paper_id, paper_reference_id, citation_context in csv_reader:
        #for record in csv_reader:
            #paper_id, paper_reference_id, citation_context = record
            record_number += 1
            solr_record = {}
            solr_record['paper_id'] = paper_id
            solr_record['paper_reference_id'] = paper_reference_id
            solr_record['citation_context'] = citation_context
            # Chunks of 500000
            if record_number % 25000 == 0:
                list_for_solr.append(solr_record)
                try:
                    solr.add(list_for_solr)
                except Exception as e:
                    print(e, record_number, filepath)
                list_for_solr = []
                print(record_number)
            else:
                list_for_solr.append(solr_record)
                #print(record_number)
        try:
            solr.add(list_for_solr)
        except Exception as e:
            print(e, record_number, filepath)

def create_concurrent_futures():
    """ Uses all the cores to do the parsing and inserting"""
    folderpath = '/home/ashwath/Files/PaperCitationContextParts/'
    refs_files = glob(os.path.join(folderpath, '*0[23].txt'))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Chunk size = 1 as we have only 10 files
        executor.map(insert_into_solr, refs_files, chunksize=1)

if __name__ == '__main__':
    start_time = time()
    create_concurrent_futures()
    print("Completed in {} seconds!".format(time() - start_time))
