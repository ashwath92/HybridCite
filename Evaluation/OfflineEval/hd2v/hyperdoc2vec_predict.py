"""1. Load the model 
2. Query 2019 papers with contexts, titles and abstracts
3. Get 100 (1000?) recommendations for each paper (start with 10) based on the hyperdoc2vec function using 
  (i) Just the contexts
  (ii) Contexts + abstract + title
4. Use the IDs to query Postgres, Get the titles back, maybe the abstracts
5. Write the IDs and abstracts into a file.
"""


"""

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
    LIMIT 10;

"""


import re
import psycopg2
import psycopg2.extras
from gensim.parsing import preprocessing
from gensim.utils import simple_preprocess
import contractions
from HyperDoc2Vec import *

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


model = HyperDoc2Vec.load('magcsenglish.model')

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

docid_prefix = '=-='
docid_suffix = '-=-'
contcounterno = 0
contcounteryes = 0
contabscounterno = 0
contabscounteryes = 0
contabstitlecounterno = 0
contabstitlecounteryes = 0
contextlengths = []
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
        outputname = '2019predictions/{}.txt'.format(paperid)
        abstractoutputname = '2019predictions/{}_withabstract.txt'.format(paperid)
        abstracttitleoutputname = '2019predictions/{}_withabstracttitle.txt'.format(paperid)
        aout = open(abstractoutputname, 'a')
        aout.write('ABSTRACT: {}\n \n'.format(abstract))
        atout = open(abstracttitleoutputname, 'a')
        atout.write('TITLE: {}\n \n'.format(title))
        atout.write('ABSTRACT: {}\n \n'.format(abstract))
        with open(outputname, 'a') as outfile:
            #outfile.write("Contexts only!!\n")
            for context in contexts:
                # consider only 1 ground truth for now.
                start_index = context.find(docid_prefix)
                end_index = context.find(docid_suffix)
                newcontext = context[:start_index] + context[end_index+3:]
                outfile.write('CONTEXT: {} \n'.format(newcontext))
                groundtruth = context[start_index+3:end_index]
                cur.execute('select papertitle from papers where paperid = {}'.format(groundtruth))
                groundtruthtitle = cur.fetchone()['papertitle']
                outfile.write('Ground Truth: {}: {}\n'.format(groundtruth, groundtruthtitle))

                # Prediction using just context
                parts = utils.to_unicode(newcontext).split()
                # pred docs is a list of tuples
                contextlengths.append(len(parts))
                pred_docs = model.predict_output_doc(parts, topn=1000)
                #Get only ids
                pred_docs_list = [doc[0] for doc in pred_docs]
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
                    contcounterno += 1
                else:
                    contcounteryes += 1
                foundornot = 'NO!\n' if foundornot == -1 else 'YES! Found at position {}\n'.format(foundornot + 1) 

                outfile.write(foundornot)
                #print(pred_docs)

                # Prediction using abstract and context
                
                contextwithabstract = '{} {}'.format(newcontext, abstract)
                aout.write('CONTEXT: {}\n'.format(newcontext))
                aout.write('Ground Truth: {}: {}\n'.format(groundtruth, groundtruthtitle))
                parts = utils.to_unicode(contextwithabstract).split()
                # pred docs is a list of tuples
                pred_docs = model.predict_output_doc(parts, topn=1000)
                #Get only ids
                pred_docs_list = [doc[0] for doc in pred_docs]
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
                    contabscounterno += 1
                else:
                    contabscounteryes += 1
                foundornot = 'NO!\n' if foundornot == -1 else 'YES! Found at position {}\n'.format(foundornot + 1)    
                aout.write(foundornot)                
                #print(pred_docs)
                # Context with abstract and title
                contextwithabstracttitle = '{} {} {}'.format(newcontext, abstract, title)
                atout.write('CONTEXT: {}\n'.format(newcontext))
                atout.write('Ground Truth: {}: {}\n'.format(groundtruth, groundtruthtitle))
                parts = utils.to_unicode(contextwithabstracttitle).split()
                # pred docs is a list of tuples
                pred_docs = model.predict_output_doc(parts, topn=1000)
                #Get only ids
                pred_docs_list = [doc[0] for doc in pred_docs]
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
                    contabstitlecounterno += 1
                else:
                    contabstitlecounteryes += 1
                foundornot = 'NO!\n' if foundornot == -1 else 'YES! Found at position {}\n'.format(foundornot + 1)    
                atout.write(foundornot)          
        aout.close()
        atout.close()
        with open('2019predictions/stats500.txt', 'w') as statsfile:
            statsfile.write('''Only context no: {}\n Only context yes: {}\n Context+abstract no:{}\n
                Context+abstract yes:{}\n Context+abstract+title no:{}\n Context+abstract+title yes:{}'''\
                .format(contcounterno, contcounteryes, contabscounterno, contabscounteryes,
                        contabstitlecounterno, contabstitlecounteryes))
#text = "kernel similarities to represent face image and evaluated their prototype random subspace (P-RS) approach on four HFR scenarios. Recently, a number of composite sketch recognition methods [1], [27], [28], [29] were proposed. Mittal et al. [29] presented a transfer learning-based deep learning representation for composite sketch recognition. Considering the insufﬁcient usage of facial spatial informat"

print(contextlengths)
import numpy as np
print("Average number of words in contexts (excluding stop words):{}".format(np.mean(contextlengths)))
#text = clean_text(text)
#parts = utils.to_unicode(text).split()
#pred_docs = model.predict_output_doc(parts, topn=100)
