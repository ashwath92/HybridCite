# -*- coding: utf-8 -*-
"""
    #-------------------------------------------------------------------------------
    # Name:        PARSE ARXIV XML
    # Purpose:     Parses the CS XML metadata files from ArXiv, and inserts some
    #              of the fields into Solr for each record.
    #
    # Author:      Ashwath Sampath
    #
    # Created:     08-03-2019
    
    #-------------------------------------------------------------------------------

"""
from collections import defaultdict
from lxml import etree
from glob import glob
import os
from time import time
import concurrent.futures
from bs4 import BeautifulSoup
from tqdm import tqdm
import pysolr
solr = pysolr.Solr('http://localhost:8983/solr/arxiv_cs_metadata', always_commit=True)

def parse_xml_insert_into_solr(filepath):
    """ Function which parses the arxiv xml for a file, and inserts some of the metadata
    into an index in Apache Solr."""
    # Set the 2 namespaces which are used in the xml file: Open archive,
    # and Dublin Core.
    namespace = {'dc': 'http://purl.org/dc/elements/1.1/',
                 'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'}
    # NOTE: this is the fully qualified version of descending through the ns
    # for m in root.find('./record/metadata/'
    # '{http://www.openarchives.org/OAI/2.0/oai_dc/}dc/'
    #'[{http://purl.org/dc/elements/1.1/}identifier=
    # "http://arxiv.org/abs/0704.0002"]'):
    
    # Make the soup
    with open(filepath) as xml:
       soup = BeautifulSoup(xml, 'lxml-xml')
    #print(soup)

    # Descend down to the metadata node in all the records: it has all the
    # interesting fields as its children.
    metadata = soup.find_all('metadata')
    #print(metadata[0])
    # metadata is a list, len(metadata) = 1000 for each file.
    list_for_solr = []
    #print(metadata)
    for metadata_record in metadata:
        solr_record = {}
        title = metadata_record.find('dc:title').string
        #print(title)
        authors_list = metadata_record.find_all('dc:creator')
        # Get the author text and discard the tags
        authors_list = [creator_tag.string for creator_tag in authors_list]
        
        # Put authors into a semicolon-separated string
        authors = ';'.join(authors_list)
        # Only get the first date, forget about the revision dates
        published_date = metadata_record.find('dc:date').string
        # find().string gets the URL here, split it and get the id (last part of url)
        arxiv_identifier = metadata_record.find('dc:identifier').string.split('/')[-1]
        solr_record['arxiv_identifier'] = arxiv_identifier
        solr_record['title'] = title
        solr_record['authors'] = authors
        solr_record['published_date'] = published_date 
        list_for_solr.append(solr_record)
    solr.add(list_for_solr)
    print("Added {} records into Solr from file {}.".format(len(list_for_solr), filepath))

def create_concurrent_futures():
    """ Uses all the cores to do the parsing and inserting"""
    folderpath = '/home/ashwath/Programs/ArxivCS/Metadata/'
    file_no = 0
    xml_files = glob(os.path.join(folderpath, '*.xml'))
    #print(xml_files)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(parse_xml_insert_into_solr, xml_files)

if __name__ == '__main__':

    start_time = time()
    create_concurrent_futures()
    print("Completed in {} seconds!".format(time() - start_time))