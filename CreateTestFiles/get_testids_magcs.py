
import psycopg2
import psycopg2.extras
import pickle
import pandas as pd
conn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=localhost")
#cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

query = """
SELECT englishcsabstracts.paperid
 FROM
    (
     SELECT englishcspapers.paperid, englishcspapers.papertitle, abstracts.abstract from
        (
            SELECT papertitle, computersciencepapers.paperid from 
            (
                SELECT papers.paperid, papertitle FROM papers 
                 INNER JOIN 
                (SELECT paperid from paperfieldsofstudy WHERE fieldofstudyid=41008148) AS fieldsofstudy 
                 ON papers.paperid=fieldsofstudy.paperid where papers.publishedyear in (2018, 2019)
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

testids = set(pd.read_sql_query(query, conn).paperid.tolist())

with open('testmagids.pickle', 'wb') as picc2:
    pickle.dump(testids, picc2)
