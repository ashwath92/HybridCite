""" Extract computer science refererences from MAG PaperReferences file (references which are present in the 
reference id column but not in the paper id column are outside the dataset, and will be discarded."""

import psycopg2
import psycopg2.extras
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('postgresql://mag:1maG$@localhost:5432/MAG19')
#conn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=localhost")
# Query to get the title, abstract and paper id together for one field of study (computer science)
# Join made on 3 tables.
query = """
SELECT paperreferences.paperid, paperreferences.paperreferenceid from 
(SELECT papers.paperid from paperfieldsofstudy pfs, papers, paperlanguages as pl where
pfs.paperid=papers.paperid and papers.paperid=pl.paperid and fieldofstudyid=41008148 and languagecode='en' and papers.citationcount>50
) as paperfields
  inner join paperreferences on paperfields.paperid = paperreferences.paperid; 
"""

from gensim.models.doc2vec import Doc2Vec
# Read the doc2vec model.
model = Doc2Vec.load('/home/ashwath/Programs/MAGCS/MAG-CS-Paper2Vec/owncontexts_p2v/MAG50CompScienced2v.dat')

#cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#cur.execute(query)
# REMEMBER: references CANNOT be OUT of MAG (they will have an id), but it's possible that the reference might
# be out of the Computer Science domain or that it might not be in English.
df = pd.read_sql_query(query, con=engine)
print(df.head())
print(df.shape)
# Keep only references which are in the dataset, i.e. only CITED papers which are also CITING papers.
# Papers which cite other papers but are not cited are retained. The inverse paperid.isin(paperreferenceid)
# will get rid of those rows. We don't want that.
df = df[df.paperreferenceid.isin(df.paperid)]
print(df.shape)
inter1 = df[df.paperid.isin(model.docvecs.offset2doctag)]
inter2 = inter1[inter1.paperreferenceid.isin(model.docvecs.offset2doctag)]
inter2.to_csv('/home/ashwath/Programs/MAGCS/MAG-CS-Paper2Vec/paperrefs_cs_en_50citations_23may.tsv', sep='\t', index=False, header=None)

# 3460025 reduced to 1048831