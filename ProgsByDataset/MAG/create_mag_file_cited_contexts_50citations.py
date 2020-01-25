import psycopg2
import psycopg2.extras
import gensim
from tqdm import tqdm
from gensim.parsing import preprocessing

file = open('/home/ashwath/Programs/MAGCS/AllYearsFiles/mag_cs_50_cited_allyears.txt', 'w')
# Do the database stuff in the global scope so that it can be read inside the class.
# Only keep the result of the cursor execution in the class __init__
conn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=localhost")
# Query to get the title, abstract and paper id together for one field of study (computer science)
# Join made on 3 tables.
query = """
SELECT englishfields.paperid, englishfields.papertitle, abstracts.abstract FROM 
(
    SELECT titlefields.paperid, titlefields.papertitle FROM 
    (
        SELECT papers.paperid, papertitle FROM papers
        INNER JOIN 
        (
            SELECT paperid FROM paperfieldsofstudy WHERE fieldofstudyid=41008148
        ) AS paperfields 
        ON paperfields.paperid=papers.paperid where papers.citationcount>50) AS titlefields 
    INNER JOIN 
    (
        SELECT paperid FROM paperlanguages WHERE languagecode='en'
    ) AS languages   
    ON titlefields.paperid=languages.paperid) AS englishfields
 INNER JOIN (select paperid, abstract FROM paperabstracts) AS abstracts
 ON abstracts.paperid=englishfields.paperid;
"""
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
second_cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

cur.execute(query)

for row in tqdm(cur):
    cur_paper_id = row.get('paperid')
    # we can now access the columns: row is a psycopg2.extras.RealDictRow (inherited from dict)
    # print(row.keys()): dict_keys(['paperid', 'papertitle', 'abstract'])
    # IMPORTANT: EXPERIMENTAL: Get the contexts from the papers which cite the current paper
    # it doesn't matter if these papers have been cited 50 times. they supply only cit contexts
    contexts_query =  """ 
      SELECT paperreferenceid, string_agg(citationcontext, ' ||--|| ') AS contexts
      FROM papercitationcontexts
      WHERE paperreferenceid=%s GROUP BY paperreferenceid; """
    second_cur.execute(contexts_query, (cur_paper_id,))
    second_results = second_cur.fetchone()
    if not second_results:
        # second_results returned None, this paper has not been cited in any citation context
        continue
    contexts = second_results['contexts']
    #contexts = contexts.split(' ||--|| ')
    contexts = preprocessing.strip_multiple_whitespaces(preprocessing.strip_non_alphanum(contexts))
    #print(contexts)
    combined_text = '{} {} {} {}\n'.format(cur_paper_id, row['papertitle'], row['abstract'], contexts)
    file.write(combined_text)
    
file.close()