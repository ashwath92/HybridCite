
import os
import pysolr
import requests
import csv


def check_if_int(volume, first_page, last_page,):
    """ volume, first_page and last_page are all str. If they are of length 0, it's okay. If they are longer than len 0, we need
    to check if they can be converted to an integer. If at least one of them can't, return False. """
    # Init 3 vars to False
    volume_good = False
    first_page_good = False
    last_page_good = False
    try: 
        # Just a dummy int(volume) statement. I just want it to give an error if it is not an integer. It will always be satisfied
        # if it is an integer
        if len(volume) > 0 and int(volume) > -999:
            volume_good = True
        else:
            volume_good = True

        # Repeat the same for first page and then last page
        if len(first_page) > 0 and int(first_page) > -999:
            first_page_good = True
        else:
            first_page_good = True

        if len(last_page) > 0 and int(last_page) > -999:
            last_page_good = True
        else:
            last_page_good = True

        if all([first_page_good, last_page_good, volume_good]):
            # ALL TRUE: return True
            return True

    except ValueError as e:
        # One of them wasn't an integer or a blank value: return False.
        # print(e)
        return False

def insert_into_solr():
    """ Inserts records into an empty solr index which has already been created."""
    solr = pysolr.Solr('http://localhost:8983/solr/mag_papers', always_commit=True)
    filepath = '/vol1/mag/data/2018-07-19/dumps/Papers.txt'
    list_for_solr = []
    record_number = 0
    # Open a rejected records file in append mode.
    rejectfile = open('/home/ashwath/Programs/IndexingMAG/RejectedRecords/papers_rejected_records.txt', 'a')
    with open(filepath, "r") as file:
        csv_reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
        #for paper_id, rank, doi, doc_type, normalized_title, original_title, book_title, publication_year, publication_date, \
        #        publisher, journal_id, conference_series_id, conference_instance_id, volume, issue, first_page, last_page, \
        #        reference_count, citation_count, estimated_citation_count, created_date in csv_reader:
        #    print(paper_id, rank, doi, doc_type, normalized_title, original_title, book_title, publication_year, publication_date, \
        #        publisher, journal_id, conference_series_id, conference_instance_id, len(volume), type(issue), len(first_page), len(last_page), \
        #        reference_count, citation_count, estimated_citation_count, created_date)
        for record in csv_reader:
            record_number += 1
            if len(record) < 21:
                # Too few fields 
                errorrecord = 'BAD RECORD: TOO FEW FIELDS ({} instead of 21)!! paper_id = {}\n'.format(len(record), record[0])
                rejectfile.write(errorrecord)
                continue

            paper_id, rank, doi, doc_type, normalized_title, original_title, book_title, publication_year, publication_date, \
            publisher, journal_id, conference_series_id, conference_instance_id, volume, issue, first_page, last_page, \
            reference_count, citation_count, estimated_citation_count, created_date = record
            #if first_page == "140917055314003":
            #    print('paper_id= {}, publisher = {}, journal_id={}, conference_series_id={}, conference_instance_id={}, volume={}, issue={}, first_page={}, last_page={}, \
            #reference_count={}, citation_count={}, estimated_citation_count={}, created_date={}'.format(paper_id, publisher, journal_id, conference_series_id,\
            # conference_instance_id, volume, issue, first_page, last_page, \
            #reference_count, citation_count, estimated_citation_count, created_date))
            #print(record, len(record), type(record))
            # Check if first page, last page and volume are integers. If not reject the record, and insert into a file rejected_records
            if not check_if_int(volume, first_page, last_page):
                # Append to rejected records file
                errorrecord = "BAD RECORD: One of first page/last page/volume NEITHER AN INTEGER NOR NULL!! paper id = {}, original title = {}, issue = {}, first_page = {}, last_page = {} \n".\
                    format(paper_id, original_title, volume, issue, first_page, last_page)
                rejectfile.write(errorrecord)
                continue
            # NOW ANOTHER ERROR HAS TO BE CHECKED. Sometimes firstpage is incorrectly taking values like 140917055314003. Check this for all. A reasonable check
            # seems to be len(first_page) > 8 (and same for the other 2 fields). I can't see any journals having a value with more than 8 digits
            if len(volume) > 8 or len(first_page) > 8 or len(last_page) > 8:
                errorrecord = "BAD RECORD: One of first page/last page/volume IS IMPOSSIBLY LARGE: outside int range!! paper id = {}, original title = {}, issue = {}, first_page = {}, last_page = {} \n".\
                    format(paper_id, original_title, volume, issue, first_page, last_page)
                rejectfile.write(errorrecord)
                continue
            
            solr_record = {}
            solr_record['paper_id'] = paper_id
            solr_record['rank'] = rank
            solr_record['doi'] = doi
            solr_record['doc_type'] = doc_type
            solr_record['normalized_title'] = normalized_title
            solr_record['original_title'] = original_title
            solr_record['book_title'] = book_title
            solr_record['publication_year'] = publication_year
            solr_record['publication_date'] = publication_date
            solr_record['publisher'] = publisher
            solr_record['journal_id'] = journal_id
            solr_record['conference_series_id'] = conference_series_id
            solr_record['conference_instance_id'] = conference_instance_id
            solr_record['volume'] = volume
            solr_record['issue'] = issue
            solr_record['first_page'] = first_page
            solr_record['last_page'] = last_page
            solr_record['reference_count'] = reference_count
            solr_record['citation_count'] = citation_count
            solr_record['estimated_citation_count'] = estimated_citation_count
            solr_record['created_date'] = created_date
            # Send chunks of 25000 to Solr.
            if record_number % 25000 == 0:
                list_for_solr.append(solr_record)
                solr.add(list_for_solr)
                list_for_solr = []
                print(record_number)
            else:
                list_for_solr.append(solr_record)

        # Upload to Solr: chunk by chunk
        solr.add(list_for_solr)
    rejectfile.close()


if __name__ == '__main__':
    insert_into_solr()
