import re
import psycopg2
import psycopg2.extras
from gensim.parsing import preprocessing
from gensim.utils import simple_preprocess
import contractions

def clean_text(text):
    """ Cleans the text in the only argument in various steps 
    ARGUMENTS: text: content/title, string
    RETURNS: cleaned text, string"""
    # Replace newlines by space. We want only one doc vector.
    text = text.replace('\n', ' ').lower()
    # Remove URLs
    #text = re.sub(r"http\S+", "", text)
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Remove stop words
    #text = preprocessing.remove_stopwords(text)
    
    #text = preprocessing.strip_tags(text)
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
    return text

# Training set: there are 220 years of data. 
# Connect to Postgres
conn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=localhost")
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

file = open('input/mag_training_data.txt', 'a')

# Computer science English papers only: title, abstract and concatenated citation contexts make up the text for each document.

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
                 ON papers.paperid=fieldsofstudy.paperid where papers.publishedyear not in (2018,2019)
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

cur.execute(query)
print("test")
docid_prefix = '=-='
docid_suffix = '-=-'
print(docid_suffix)
for row in cur:
    # row is a dict with keys:
    # dict_keys(['paperid', 'papertitle', 'abstract', 'contexts', 'referenceids'])
    paperid = row.get('paperid')
    contexts = row.get('contexts')
    referenceids = row.get('referenceids')
    title = clean_text(row.get('papertitle'))
    abstract = clean_text(row.get('abstract'))
    #title = row.get('papertitle')
    #abstract = row.get('abstract')

    print(title)

    # Get a single string for all the contexts
    if contexts is not None and referenceids is not None:
        contexts = contexts.split(' ||--|| ')
        referenceids = referenceids.split(',')
        contexts_with_refs = []
        for context, referenceid in zip(contexts, referenceids):
            # Clean text and split into a list
            contextlist = clean_text(context).split()
            # Insert the reference id as the MIDDLE word of the context
            # NOTE, when multiple reference ids are present, only 1 is inserted. Mag issue.
            # In the eg. nips file, it's like this: this paper uses our previous work on weight space 
            # probabilities =-=nips05_0451-=- =-=nips05_0507-=-. 
            index_to_insert = len(contextlist) // 2
            value_to_insert = docid_prefix + referenceid + docid_suffix
            # Add the ref id with the prefix and suffix
            contextlist.insert(index_to_insert, value_to_insert)
            contexts_with_refs.append(' '.join(contextlist))
            contexts_concatenated = ' '.join(contexts_with_refs)

    else:
        contexts_concatenated = ''

    # Concatenate the paperid, title, abstract and the contexts together.
    content = "{} {} {} {}\n".format(paperid, title, abstract, contexts_concatenated)
    print(content)
    file.write(content)
    print("INSERTED LINE")
file.close()

