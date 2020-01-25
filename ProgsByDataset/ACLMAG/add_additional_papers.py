import pickle
from tqdm import tqdm
from gensim.parsing import preprocessing
from gensim.utils import to_unicode
import contractions
import psycopg2
import psycopg2.extras


with open('Pickles/allmagpapers_en_magcontexts.pickle', 'rb') as allpickle:
    allmagpaperids = pickle.load(allpickle)

with open('Pickles/inacl_papers_set.pickle', 'rb') as inaclpickle:
    inacl_papers_set = pickle.load(inaclpickle)

outfile = open('acl_training_data.txt', 'a')
docid_prefix='=-='
docid_suffix='-=-'

# This is a dict writer in the original program
acl_citing_cited_file = open('AdditionalOutputs/aclmag_references.tsv', 'a')
acl_citing_cited_file.write('\n')

pconn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=shetland.informatik.uni-freiburg.de")
pcur = pconn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# POSTGRES QUERY
magonly_query = """
SELECT titleandabstract.paperid, papertitle, abstract, contexts, referenceids
FROM  
    (
        SELECT papers.paperid, papertitle, abstract FROM papers INNER JOIN paperabstracts 
        ON papers.paperid=paperabstracts.paperid
        WHERE papers.paperid=%s) AS titleandabstract INNER JOIN 
        (
            SELECT paperid, string_agg(paperreferenceid::character varying, ',') AS referenceids,
            string_agg(citationcontext, ' ||--|| ') AS contexts 
            FROM papercitationcontexts 
            WHERE paperid=%s 
            GROUP BY paperid
        ) AS listofcontexts
        ON titleandabstract.paperid=listofcontexts.paperid;"""

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


def add_additional_papers():
    """ Add additional papers for which full text from ACL is not present. Care is taken that while
    adding references to THESE papers, these references should be in the set of papers stored
    in the allmagpaperids set (otherwise, there will be additional papers in the reference part
    of the concat contexts which are not in the files in the text.
    ALSO NOTE that allmagpaperids contains all papers which either cite or are cited so far
    inacl_papers_set contains the set of papers which are in acl (citing)
    A set difference (allmagpaperids - inacl_papers_set) gives the set of mag_ids for which we 
    get additional text"""

    additional_mag_ids = allmagpaperids - inacl_papers_set
    for paperid in tqdm(additional_mag_ids):
        pcur.execute(magonly_query, (paperid, paperid))
        # Get paperid, contexts, abstract, title, refids of current paper id
        for row in pcur:
            # row is a dict with keys:
            # dict_keys(['paperid', 'papertitle', 'abstract', 'contexts', 'referenceids'])
            paperid = row.get('paperid')
            # Get all contexts and reference ids (delimiters set in the pSQL query)
            contexts = row.get('contexts').replace('\n', ' ')
            referenceids = row.get('referenceids')
            title = clean_text(row.get('papertitle'))
            abstract = clean_text(row.get('abstract'))
            print(title)
            # Get a single string for all the contexts
            if contexts is not None and referenceids is not None:
                contexts = contexts.split(' ||--|| ')
                referenceids = referenceids.split(',')
                contexts_with_refs = []
                # Go through context, refid pairs, one at a time
                for context, referenceid in zip(contexts, referenceids):
                    # VERY VERY IMPORTANT: check if the referenceid is not present in the allmagpaperids set,
                    # IGNORE IT! DESIGN DECISION: the other choice is to have a LOT of passes. 
                    if referenceid in allmagpaperids:
                        acl_citing_cited_file.write('{}\t{}\n'.format(paperid, referenceid))
                        #writer.writerow({'citing_mag_id': paperid,'cited_mag_id': referenceid})
                        contextlist = clean_text(context).split()
                        # Insert the reference id as the MIDDLE word of the context
                        # NOTE, when multiple reference ids are present, only 1 is inserted. Mag issue.
                        # In the eg. nips file, it's like this: this paper uses our previous work on weight space 
                        # probabilities =-=nips05_0451-=- =-=nips05_0507-=-. 
                        index_to_insert = len(contextlist) // 2
                        value_to_insert = docid_prefix + referenceid + docid_suffix
                        # Add the ref id with the prefix and suffix
                        contextlist.insert(index_to_insert, value_to_insert)
                        # Insert the context with ref id into the contexts_with_refs list
                        contexts_with_refs.append(' '.join(contextlist))
                    # else: do nothing, next iteration
                # After all the contexts azre iterated to, make them a string.
                contexts_concatenated = ' '.join(contexts_with_refs)                    
            else:
                contexts_concatenated = ''
                # Do not write these to file????? OR 
            # Concatenate the paperid, title, abstract and the contexts together.
            content = "{} {} {} {}\n".format(paperid, title, abstract, contexts_concatenated)
            content = to_unicode(content)
            if content.strip() != '':
                outfile.write(content)
                print("Written in file for {}".format(paperid))

if __name__ == '__main__':
    add_additional_papers()
    outfile.close()
    acl_citing_cited_file.close()