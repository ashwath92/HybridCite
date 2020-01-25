import psycopg2
import psycopg2.extras
import gensim
from tqdm import tqdm

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
        ON paperfields.paperid=papers.paperid where publishedyear not in (2018,2019)) AS titlefields 
    INNER JOIN 
    (
        SELECT paperid FROM paperlanguages WHERE languagecode='en'
    ) AS languages   
    ON titlefields.paperid=languages.paperid) AS englishfields
 INNER JOIN (select paperid, abstract FROM paperabstracts) AS abstracts
 ON abstracts.paperid=englishfields.paperid;
"""

conn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=localhost")
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Write the document vectors to file
def write():
    model = gensim.models.Doc2Vec.load('MAGCompScienced2v.dat')
    out_f = open('MAGCompScienced2v.txt', 'w')
    # The first row is the number of lines (i.e. len(model.docvecs.doctags), and the size of the vector)
    numdocvecs = len(model.docvecs)
    vector_size = model.vector_size
    out_f.write('{} {}\n'.format(numdocvecs, vector_size))

    cur.execute(query)
    for row in cur:
        paperid = repr(row['paperid'])
        try:
            vect = model.docvecs[paperid]
        except KeyError:
            continue
        str_vect = ' '.join([str(j) for j in vect])
        string = "{} {}".format(paperid, str_vect) 
        out_f.write(string + '\n')
    out_f.close()


if __name__ == '__main__':
    write()
