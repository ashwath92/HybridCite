
import csv
import os
import requests
from glob import glob
import concurrent.futures

def download_from_url(filepath):
    """ Downloads the pdf from the pdf URL obtained in a delimited file with paper id (MAG), doi and pdf url
    using the requests module. The names of the pdf are MAG paper ids""" 
    #filepath = '/home/ashwath/Files/Unpaywall/Parts/{}'.format(filename)
    input_filename = os.path.basename(filepath)
    output_filepath = "/home/ashwath/Files/Unpaywall/Fulltext/pdfs/"
    # Create the ouptut path if it doesn't exist
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    # Create a file which contains the pdfs which could not be downloaded. 
    fieldnames = ['paper_id', 'doi', 'pdf_url', 'exception']
    notfound_tsv = open('/home/ashwath/Files/Unpaywall/Fulltext/NotFound/pdf_not_found{}'.format(input_filename), 'w')
    notfound_writer = csv.DictWriter(notfound_tsv, delimiter='\t', fieldnames=fieldnames)
    #notfound_writer.writeheader()
    
    with open(filepath, 'r') as input_file:
        csv_reader = csv.reader(input_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        # IMPORTANT: the file has a header, this needs to be skipped
        next(csv_reader, None)   
        for paper_id, doi, pdf_url in csv_reader:
            # Get a stream response from the PDF_URL. This will be iterated and written as bytes to the output
            # file
            try:
                # VERY IMPORTANT: Add User Agent in the Content header. When a page redirects, it produces a ConnectionError.
                # Adding the user agent prevents this and gets the response from the redirected page (allow_redirects=True by default)
                # https://stackoverflow.com/questions/36749376/python-issues-with-httplib-requests-https-seems-to-cause-a-redirect-then-bad
                user_agent_header = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36"}
                url_response = requests.get(pdf_url, stream=True, timeout=15, headers=user_agent_header)
                # print(url_response.url) -> redirected url
                filename = '{output_filepath}{mag_paper_id}.pdf'.format(output_filepath=output_filepath, mag_paper_id=paper_id)
            
                with open(filename, 'wb') as pdf_file:
                    # As the file may be large, write to it in chunks of 2000 bytes
                    for chunk in url_response.iter_content(chunk_size=2000):
                        pdf_file.write(chunk)
            except requests.exceptions.ConnectionError as ce:
                url_response = "No response"
                notfound_writer.writerow({'paper_id': paper_id, 'doi': doi, 'pdf_url': pdf_url, 'exception': ce})
                print('{} ({}) not found'.format(paper_id, pdf_url))
                # Go to the next line in the input file if the pdf is not found
                continue

            except requests.exceptions.Timeout as te:
                url_response = "Timeout"
                notfound_writer.writerow({'paper_id': paper_id, 'doi': doi, 'pdf_url': pdf_url, 'exception': te})
                print('{} ({}) not found'.format(paper_id, pdf_url))
                # Go to the next line in the input file if the pdf is not found
                continue

            except requests.exceptions.RequestException as e:
                # Catch all other exceptions
                url_response = "No response"
                notfound_writer.writerow({'paper_id': paper_id, 'doi': doi, 'pdf_url': pdf_url, 'exception': e})
                print('{} ({}) not found'.format(paper_id, pdf_url))
                # Go to the next line in the input file if the pdf is not found
                continue
    notfound_writer.close()

def create_concurrent_futures():
    """ Uses all the cores to do the parsing and inserting"""
    folderpath = '/home/ashwath/Files/Unpaywall/Parts/'
    files = glob(os.path.join(folderpath, '*.tsv'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        # chunk size =1
        executor.map(download_from_url, files)


if __name__ == '__main__':
    create_concurrent_futures()
