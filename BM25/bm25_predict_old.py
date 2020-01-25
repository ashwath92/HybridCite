"""1. Load the model 
2. Query 2019 papers with contexts, titles and abstracts
3. Get 100 (1000?) recommendations for each paper (start with 10) based on the hyperdoc2vec function using 
  (i) Just the contexts
  (ii) Contexts + abstract + title
4. Use the IDs to query Postgres, Get the titles back, maybe the abstracts
5. Write the IDs and abstracts into a file.
"""


import re
import psycopg2
import psycopg2.extras
from gensim.parsing import preprocessing
from gensim.utils import to_unicode
from gensim.utils import simple_preprocess
import contractions
import requests
import pysolr


query = """
SELECT englishcsabstracts.paperid, englishcsabstracts.papertitle, englishcsabstracts.abstract, citationcontexts.contexts, citationcontexts.referenceids
 FROM
    (
     SELECT englishcspapers.paperid, englishcspapers.papertitle, abstracts.abstract from
        (
            SELECT papertitle, computersciencepapers.paperid from 
            (
                SELECT papers.paperid, papertitle FROM papers 
                 INNER JOIN 
                (SELECT paperid from paperfieldsofstudy WHERE fieldofstudyid=41008148) AS fieldsofstudy 
                 ON papers.paperid=fieldsofstudy.paperid where papers.publishedyear=2019
            ) AS computersciencepapers
            INNER JOIN 
            (SELECT paperid FROM paperlanguages WHERE languagecode='en') AS languages 
            ON languages.paperid=computersciencepapers.paperid
        ) AS englishcspapers INNER JOIN 
        (SELECT paperid, abstract FROM paperabstracts) AS abstracts
        ON abstracts.paperid=englishcspapers.paperid
    ) AS englishcsabstracts INNER JOIN 
    (SELECT paperid, string_agg(paperreferenceid::character varying, ',') AS referenceids, string_agg(citationcontext, ' ||--|| ') AS contexts
    FROM papercitationcontexts GROUP BY paperid) AS citationcontexts 
    ON citationcontexts.paperid=englishcsabstracts.paperid; """




conn = psycopg2.connect("dbname=MAG user=mag password=1maG$ host=localhost")
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def clean_text(text):
    """ Cleans the text in the only argument in various steps 
    ARGUMENTS: text: content/title, string
    RETURNS: cleaned text, string"""
    # Replace newlines by space. We want only one doc vector.
    text = text.replace('\n', ' ').lower()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Remove stop words
    text = preprocessing.remove_stopwords(text)
    # Remove html tags and numbers: can numbers possible be useful?
    text = preprocessing.strip_tags(text)
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
    return text

def search_solr_parse_json(query, collection, search_fields):
    """ Searches the nounphrases collection on query,
    parses the json result and returns it as a list of dictionaries where
    each dictionary corresponds to a record. 
    ARGUMENTS: query, string: the user's query entered in a search box
               (if it is comma-separated, only one part of the query is sent
               to this function).
               collection: the Solr collection name (=nounphrases)
               search_field: the Solr field which is queried (=phrase)
    RETURNS: docs, list of dicts: the documents (records) returned by Solr 
             AFTER getting the JSON response and parsing it."""
    solr_url = 'http://localhost:8983/solr/' + collection + '/select'
    # Exact search only
    #query = '"' + query + '"'
    # for rows, pass an arbitrarily large number.
    url_params = {'defType': 'dismax', 'q': query, 'rows': 1000, 'qf': search_fields}
    solr_response = requests.get(solr_url, params=url_params)
    if solr_response.ok:
        data = solr_response.json()
        docs = data['response']['docs']
        return docs
    else:
        print("Invalid response returned from Solr")
        sys.exit(11)

docid_prefix = '=-='
docid_suffix = '-=-'
cont_titabs_counterno = 0
cont_titabs_counteryes = 0
cont_titabscont_counterno = 0
cont_titabscont_counteryes = 0
contabs_titabs_counterno = 0
contabs_titabs_counteryes = 0
contabs_titabscont_counterno = 0
contabs_titabscont_counteryes = 0
contabstit_titabs_counterno = 0
contabstit_titabs_counteryes = 0
contabstit_titabscont_counterno = 0
contabstit_titabscont_counteryes = 0

# File format: paperid title $$&&$$ abstract @@&&@@ contexts separated by ||--||
with open('mag_2019_csen.txt', 'r') as file:
    for line in file:
        content = line.split()
        paperid = content[0]
        content = ' '.join(content[1:])
        parts = content.split(' $$&&$$ ')
        title = parts[0]
        title = clean_text(title)
        content = parts[1]
        parts = content.split(' @@&&@@ ')
        abstract = parts[0]
        abstract = clean_text(abstract)
        contexts = parts[1].split(' ||--|| ')
        #print(contexts)
        outputname = '2019predictions/{}_context_with_titleabstract.txt'.format(paperid)
        secondoutput = '2019predictions/{}_context_with_titleabstractcontexts.txt'.format(paperid)
        abstractoutputname = '2019predictions/{}_contextabstract_with_titleabstract.txt'.format(paperid)
        secondabstractoutput = '2019predictions/{}_contextabstract_with_titleabstractcontexts.txt'.format(paperid)
        abstracttitleoutputname = '2019predictions/{}_contexttitleabstract_with_abstracttitle.txt'.format(paperid)
        secondabstracttitleoutput = '2019predictions/{}_contexttitleabstract_with_abstracttitlecontexts.txt'.format(paperid)
        aout = open(abstractoutputname, 'a')
        aout.write('ABSTRACT: {}\n \n'.format(abstract))
        aout2 = open(secondabstractoutput, 'a')
        aout2.write('ABSTRACT: {}\n \n'.format(abstract))
        atout = open(abstracttitleoutputname, 'a')
        atout.write('TITLE: {}\n \n'.format(title))
        atout.write('ABSTRACT: {}\n \n'.format(abstract))
        atout2 = open(secondabstracttitleoutput, 'a')
        atout2.write('TITLE: {}\n \n'.format(title))
        atout2.write('ABSTRACT: {}\n \n'.format(abstract))

        with open(outputname, 'a') as outfile, open(secondoutput, 'a') as secondoutputfile:
            #outfile.write("Contexts only!!\n")
            for context in contexts:
                # consider only 1 ground truth for now.
                start_index = context.find(docid_prefix)
                end_index = context.find(docid_suffix)
                newcontext = context[:start_index] + context[end_index+3:]
                outfile.write('CONTEXT: {} \n'.format(newcontext))
                secondoutputfile.write('CONTEXT: {} \n'.format(newcontext))
                groundtruth = context[start_index+3:end_index]
                cur.execute('select papertitle from papers where paperid = {}'.format(groundtruth))
                groundtruthtitle = cur.fetchone()['papertitle']
                outfile.write('Ground Truth: {}: {}\n'.format(groundtruth, groundtruthtitle))
                secondoutputfile.write('Ground Truth: {}: {}\n'.format(groundtruth, groundtruthtitle))

                # Prediction by querying just context of data on titleabstract in solr
                newcontext = to_unicode(newcontext)
                pred_docs_dict = search_solr_parse_json(newcontext, 'mag_en_cs', 'titleabstract')
                #Get only ids
                #print(pred_docs_dict)
                pred_docs_list = [doc['paperid'] for doc in pred_docs_dict]
                #print(type(pred_docs_list[0]), type(groundtruth)): both str
                pred_docs = ','.join(pred_docs_list)
                cur.execute('select paperid, papertitle from papers where paperid in ({})'.format(pred_docs))
                outfile.write('Predictions:\n')
                for row in cur:
                    outfile.write('{}: {}\n'.format(row.get('paperid'), row.get('papertitle')))
                try:
                    foundornot = pred_docs_list.index(groundtruth)
                except ValueError:
                    foundornot = -1
                if foundornot == -1:
                    cont_titabs_counterno += 1
                else:
                    cont_titabs_counteryes += 1
                foundornot = 'NO!\n' if foundornot == -1 else 'YES! Found at position {}\n'.format(foundornot + 1) 

                outfile.write(foundornot)
                
                # Context is used to search on titleabstract + contexts
                pred_docs_dict = search_solr_parse_json(newcontext, 'mag_en_cs', 'titleabstract contexts')
                # Get only ids
                # print(pred_docs_dict)
                pred_docs_list = [doc['paperid'] for doc in pred_docs_dict]
                # print(type(pred_docs_list[0]), type(groundtruth)): both str
                pred_docs = ','.join(pred_docs_list)
                cur.execute('select paperid, papertitle from papers where paperid in ({})'.format(pred_docs))
                secondoutputfile.write('Predictions:\n')
                for row in cur:
                    secondoutputfile.write('{}: {}\n'.format(row.get('paperid'), row.get('papertitle')))
                try:
                    foundornot = pred_docs_list.index(groundtruth)
                except ValueError:
                    foundornot = -1
                if foundornot == -1:
                    cont_titabscont_counterno += 1
                else:
                    cont_titabscont_counteryes += 1
                foundornot = 'NO!\n' if foundornot == -1 else 'YES! Found at position {}\n'.format(foundornot + 1) 
                secondoutputfile.write(foundornot)
                
                # Context + abstract is used to search on titleabstract
                
                contextwithabstract = '{} {}'.format(newcontext, abstract)
                aout.write('CONTEXT: {}\n'.format(newcontext))
                aout.write('Ground Truth: {}: {}\n'.format(groundtruth, groundtruthtitle))
                contextwithabstract = to_unicode(contextwithabstract)
                pred_docs = search_solr_parse_json(contextwithabstract, 'mag_en_cs', 'titleabstract')
                #Get only ids
                pred_docs_list = [doc['paperid'] for doc in pred_docs]
                pred_docs = ','.join(pred_docs_list)
                cur.execute('select paperid, papertitle from papers where paperid in ({})'.format(pred_docs))
                aout.write('Predictions:\n')
                for row in cur:
                    aout.write('{}: {}\n'.format(row.get('paperid'), row.get('papertitle')))
                try:
                    foundornot = pred_docs_list.index(groundtruth)
                except ValueError:
                    foundornot = -1
                if foundornot == -1:
                    contabs_titabs_counterno += 1
                else:
                    contabs_titabs_counteryes += 1
                foundornot = 'NO!\n' if foundornot == -1 else 'YES! Found at position {}\n'.format(foundornot + 1)    
                aout.write(foundornot)                
                #print(pred_docs)

                # Context + abstract is used to search on titleabstract+contexts

                aout2.write('CONTEXT: {}\n'.format(newcontext))
                aout2.write('Ground Truth: {}: {}\n'.format(groundtruth, groundtruthtitle))
                pred_docs = search_solr_parse_json(contextwithabstract, 'mag_en_cs', 'titleabstract contexts')
                #Get only ids
                pred_docs_list = [doc['paperid'] for doc in pred_docs]
                pred_docs = ','.join(pred_docs_list)
                cur.execute('select paperid, papertitle from papers where paperid in ({})'.format(pred_docs))
                aout.write('Predictions:\n')
                for row in cur:
                    aout2.write('{}: {}\n'.format(row.get('paperid'), row.get('papertitle')))
                try:
                    foundornot = pred_docs_list.index(groundtruth)
                except ValueError:
                    foundornot = -1
                if foundornot == -1:
                    contabs_titabscont_counterno += 1
                else:
                    contabs_titabscont_counteryes += 1
                foundornot = 'NO!\n' if foundornot == -1 else 'YES! Found at position {}\n'.format(foundornot + 1)    
                aout2.write(foundornot)                
                
                # Context + abstract + title is used to search on titleabstract
                contextwithabstracttitle = '{} {} {}'.format(newcontext, abstract, title)
                atout.write('CONTEXT: {}\n'.format(newcontext))
                atout.write('Ground Truth: {}: {}\n'.format(groundtruth, groundtruthtitle))
                contextwithabstracttitle = to_unicode(contextwithabstracttitle)
                # pred docs is a list of tuples
                pred_docs = search_solr_parse_json(contextwithabstract, 'mag_en_cs', 'titleabstract')
                #Get only ids
                pred_docs_list = [doc['paperid'] for doc in pred_docs]
                pred_docs = ','.join(pred_docs_list)
                cur.execute('select paperid, papertitle from papers where paperid in ({})'.format(pred_docs))
                atout.write('Predictions:\n')
                for row in cur:
                    atout.write('{}: {}\n'.format(row.get('paperid'), row.get('papertitle')))
                try:
                    foundornot = pred_docs_list.index(groundtruth)
                except ValueError:
                    foundornot = -1
                if foundornot == -1:
                    contabstit_titabs_counterno += 1
                else:
                    contabstit_titabs_counteryes += 1
                foundornot = 'NO!\n' if foundornot == -1 else 'YES! Found at position {}\n'.format(foundornot + 1)    
                atout.write(foundornot) 

                # Context + abstract + title is used to search on titleabstract+contexts
                atout2.write('CONTEXT: {}\n'.format(newcontext))
                atout2.write('Ground Truth: {}: {}\n'.format(groundtruth, groundtruthtitle))
                # pred docs is a list of tuples
                pred_docs = search_solr_parse_json(contextwithabstract, 'mag_en_cs', 'titleabstract contexts')
                #Get only ids
                pred_docs_list = [doc['paperid'] for doc in pred_docs]
                pred_docs = ','.join(pred_docs_list)
                cur.execute('select paperid, papertitle from papers where paperid in ({})'.format(pred_docs))
                atout.write('Predictions:\n')
                for row in cur:
                    atout2.write('{}: {}\n'.format(row.get('paperid'), row.get('papertitle')))
                try:
                    foundornot = pred_docs_list.index(groundtruth)
                except ValueError:
                    foundornot = -1
                if foundornot == -1:
                    contabstit_titabscont_counterno += 1
                else:
                    contabstit_titabscont_counteryes += 1
                foundornot = 'NO!\n' if foundornot == -1 else 'YES! Found at position {}\n'.format(foundornot + 1)    
                atout2.write(foundornot)
         
        aout.close()
        atout.close()
        aout2.close()
        atout2.close()
        with open('2019predictions/stats.txt', 'w') as statsfile:
            statsfile.write("Prediction by querying context on titleabstract in solr\n")
            statsfile.write("No: {}\n Yes: {}".format(cont_titabs_counterno, cont_titabs_counteryes)) 
            statsfile.write("Prediction by querying context on titleabstract+contexts in solr\n")
            statsfile.write("No: {}\n Yes: {}".format(cont_titabscont_counterno, cont_titabscont_counteryes)) 
            statsfile.write("Prediction by querying context+abstract on titleabstract in solr\n")
            statsfile.write("No: {}\n Yes: {}".format(contabs_titabs_counterno, contabs_titabs_counteryes)) 
            statsfile.write("Prediction by querying context+abstract on titleabstract+contexts in solr\n")
            statsfile.write("No: {}\n Yes: {}".format(contabs_titabscont_counterno, contabs_titabscont_counteryes)) 
            statsfile.write("Prediction by querying context+abstract+title on titleabstract in solr\n")
            statsfile.write("No: {}\n Yes: {}".format(contabstit_titabs_counterno, contabstit_titabs_counteryes)) 
            statsfile.write("Prediction by querying context+abstract+title on titleabstract+contexts in solr\n")
            statsfile.write("No: {}\n Yes: {}".format(contabstit_titabscont_counterno, contabstit_titabscont_counteryes)) 
