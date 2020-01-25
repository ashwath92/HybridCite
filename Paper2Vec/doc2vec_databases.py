## Soumyajit Ganguly
## soumyajit.ganguly@research.iiit.ac.in
## MS, IIIT Hyderabad

# Ashwath Sampath, 28 Nov 2018: revised to use databases and MAG

import multiprocessing
import logging
import random
import time
import os
import re
import psycopg2
import psycopg2.extras
import gensim
from tqdm import tqdm
from gensim.parsing import preprocessing
from gensim.utils import simple_preprocess
import contractions

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
        ON paperfields.paperid=papers.paperid where publishedyear not in (2018,2019)) AS titlefields 
    INNER JOIN 
    (
        SELECT paperid FROM paperlanguages WHERE languagecode='en'
    ) AS languages   
    ON titlefields.paperid=languages.paperid) AS englishfields
 INNER JOIN (select paperid, abstract FROM paperabstracts) AS abstracts
 ON abstracts.paperid=englishfields.paperid;
"""

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

# Iterate over document folder and read 1by1.
# Documents are already pre-processed by nltk tokenizer

# fetch only one row. We'll go through the cursor in the __iter__ function till it is exhausted.
# Also see this: https://stackoverflow.com/questions/17933344/python-postgres-can-i-fetchall-1-million-rows
# And this https://stackoverflow.com/questions/6195988/about-mysql-cursor-and-iterator
# And this https://stackoverflow.com/questions/6739355/dictcursor-doesnt-seem-to-work-under-psycopg2
#self.row = cur.fetchone()
class DatabaseTaggedDocument(object):
    def __init__(self, cur, second_cur):
        self.cur = cur
        self.second_cur = second_cur

    def __iter__(self):
        # psycopg2's cursor implements __iter__, so we can iterate through it to get the rows. it calls fetchone internally
        # the query execution has to be in iter and not init!! Otherwise, it will be an iterator and not an iterable,
        # and will not work over multiple epochs
        self.cur.execute(query)
        for row in tqdm(self.cur):
            cur_paper_id = row.get('paperid')
            # we can now access the columns: row is a psycopg2.extras.RealDictRow (inherited from dict)
            # print(row.keys()): dict_keys(['paperid', 'papertitle', 'abstract'])
            # IMPORTANT: EXPERIMENTAL: Get the contexts from the papers which cite the current paper
            contexts_query =  """ 
            SELECT paperreferenceid, string_agg(citationcontext, ' ||--|| ') AS contexts,
             string_agg(paperid::character varying, ',') AS paperids FROM papercitationcontexts 
             WHERE paperreferenceid=%s GROUP BY paperreferenceid; """
            self.second_cur.execute(contexts_query, (cur_paper_id,))
            second_results = self.second_cur.fetchone()
            if not second_results:
                # second_results returned None, this paper has not been cited in any citation context
                continue
            paperids = second_results['paperids']
            contexts = second_results['contexts']
            contexts = contexts.split(' ||--|| ')
            paperids = paperids.split(',')
            contexts_with_paperids = []
            for context, paperid in zip(contexts, paperids):
                contexts_with_paperids.append('{} {}'.format(paperid, context))
            contexts = ' '.join(contexts_with_paperids)
            #print(contexts)
            combined_text = row['papertitle'] + ' ' + row['abstract'] + contexts
            #print(combined_text)

            # row['paperid'] is an int
            yield gensim.models.doc2vec.TaggedDocument(words=simple_preprocess(self.clean_text(combined_text), deacc=True, min_len=2), 
                tags=[repr(row['paperid'])])

    def clean_text(self, text):
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
        # Remove html tags
        text = preprocessing.strip_tags(text)
        # Remove punctuation -- all special characters
        text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
        return text


# Write the document vectors to file
def write(model, cur):
    model = gensim.models.Doc2Vec.load('MAGCompScienced2v.dat')
    out_f = open('MAGCompScienced2v.txt', 'w')
    numdocvecs = len(model.docvecs)
    vector_size = model.vector_size
    out_f.write('{} {}\n'.format(numdocvecs, vector_size))

    #cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(query)
    for row in cur:
        paperid = repr(row['paperid'])
        vect = model.docvecs[paperid]
        str_vect = ' '.join([str(j) for j in vect])
        string = "{} {}".format(paperid, str_vect) 
        out_f.write(string + '\n')
    out_f.close()

def main():
    # dm is the default
    model = gensim.models.Doc2Vec(alpha=0.025, window=10, min_count=5, min_alpha=0.025, vector_size=500, workers=cores, sample=1e-4)  # use fixed learning rate
    # Create a DictCursor (like sqlite Row)
    dict_cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    second_cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    sentences = DatabaseTaggedDocument(dict_cur, second_cur)
    model.build_vocab(sentences)
    #model.intersect_word2vec_format('w2v_100.txt')  # pretraining with word2vec learnt on bigger corpus (not needed)
    #t1 = time.time()
    for epoch in tqdm(range(10)):
        # Added total_examples and epochs.
        model.train(sentences, total_examples=model.corpus_count, epochs=1)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    # model will have a trained doc2vec model with paperids as the Tags in the TaggedDocument, title+abstract words as the words
    model.save('MAGCompScienced2v.dat')
    third_cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    # Cancel writing, keep that to a separate program.
    #write(model, third_cur)

    # Later do infer_vector FOR A NEW DOC: same preproc, tokenizing: 
    #model.random.seed(0)
    #v = model.infer_vector(['jhju jknu'])
    # model.docvecs.most_similar([v])
    # https://github.com/RaRe-Technologies/gensim/issues/447: check this, he seems to have used the code directly from a comment.
    # seeds and infer vector detailed in the link

if __name__ == '__main__':
    main()

"""
doctext=Scaling Up Average Reward Reinforcement Learning by Approximating the Domain Models and the Value Function Almost
 all the work in Average-reward Reinforcement Learning ( ARL ) so far has focused on table-based methods which do not scale 
 to domains with large state spaces . In this paper , we propose two extensions to a model-based ARL method called H-learning
  to address the scale-up problem . We extend H-learning to learn action models and reward functions in the form of Bayesian 
  networks , and approximate its value function using local linear regression . We test our algorithms on several scheduling 
  tasks for a simulated Automatic Guided Vehicle ( AGV ) and show that they are effective in significantly reducing the space
   requirement of H-learning and making it converge faster . To the best of our knowledge , our results are the first in apply 
   ing function approximation to ARL . Recently , there has been growing interest in Average-reward Reinforcement Learning 
   ( ARL ) ( Schwartz , 1993 ; Singh , 1994 ; Mahadevan , 1996 ; Boutilier & Puterman , 1995 ; Tadepalli & Ok , 1994 ) . In
    this paper , we study two extensions to a model-based ARL method called H-learning , an undis-counted version of Adaptive 
    Real-Time Dynamic Programming ( ARTDP ) ( Barto et al. , 1995 ) . The study of ARL so far has been restricted to table-based 
    methods in which the value function is stored as a table . In the worst case , this table can become as large as the number 
    of states in the state space . It is very similar to `` Algorithm B  of Jalali and Ferguson ( Jalali & Ferguson , 1989 ) , and
     is an Average-reward version of the discounted RL method , Adaptive Real-time Dynamic Programming ( ARTDP ) ( Barto et al. ,
    1995 ) . It can also be seen as a model-based version of R-learning ( Schwartz , 1993 ) . The right hand side of Equation ( 1 )
    also involves , which is estimated as the gain of the current greedy policy . It initializes all the h values to 0 , and in 
    each current state i , updates h ( i ) with the right hand side of Equation ( 1 ) . It is very similar to `` Algorithm B  of 
    Jalali and Ferguson ( Jalali & Ferguson , 1989 ) , and is an Average-reward version of the discounted RL method , Adaptive Real-time
    Dynamic Programming ( ARTDP ) ( Barto et al. , 1995 ) . It can also be seen as a model-based version of R-learning ( Schwartz , 1993 ) 
    H-learning improves its ability to automatically explore the state space by initializing the value of to a value 0 higher than
    the expected optimum gain and adjusting it using a decreasing learning rate ff , initialized to a small value , ff 0 ( Ok & Tadepalli
    , 1996 ) . Our experiments are based on this version of H-learning , denoted by H 0 ; ff 0 . 3 Model Generalization using Bayesian
    Networks One of the disadvantages of model-based methods like H-learning is that explicitly storing its action and re ward models
    consumes a lot of space . = 6 6 fi 0 . . . 3 7 5 2 6 4 e 2 e m 7 7 : The values for parameters fi 0 ; : : : ; fi k that minimize
    the squared error can be obtained as the ( k + 1 ) fi 1 vector B ( Canavos , 1984 ) . B = ( X T X ) 1 X T Y : ( 2 ) 4.2 Local 
    Linear Regression Let us assume that the state is represented by a set of k `` linear  features and n k `` nonlinear  features 
    . Model-based methods like H-learning also have the additional problem of having to explicitly store their action models and the reward 
    functions , which we call the `` domain models .  Dynamic Bayesian networks have been successfully used in the past to represent the domain
    models ( Russell & Norvig , 1995 ) . In many cases , it is possible to design these networks in such a way that a small number of
    parameters are sufficient to fully specify the domain models .'

model = gensim.models.Doc2Vec.load('MAGCompScienced2v.dat')

tts = clean_text(doctext)
def clean_text(text):
        # Replace newlines by space. We want only one doc vector.
        text = text.replace('\n', ' ').lower()
        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        # Expand contractions: you're to you are and so on.
        text = contractions.fix(text)
        # Remove stop words
        text = preprocessing.remove_stopwords(text)
        # Remove html tags and numbers: can numbers possible be useful?
        text = preprocessing.strip_tags(preprocessing.strip_numeric(text))
        # Remove punctuation -- all special characters
        text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
        return text
new = model.infer_vector(simple_preprocess(tts))

>>> pprint(model.docvecs.most_similar([new], topn=20))
/home/ashwath/.local/lib/python3.6/site-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.
  if np.issubdtype(vec.dtype, np.int):
[('2745520940', 0.8012994527816772),
 ('2355995632', 0.7995553612709045),
 ('2390038715', 0.7968725562095642),
 ('2527614997', 0.7955911159515381),
 ('2567356174', 0.7952933311462402),
 ('642958332', 0.794701874256134),
 ('2782065452', 0.7945420742034912),
 ('2547332013', 0.7925412654876709),
 ('2404516878', 0.7920867204666138),
 ('2574095046', 0.7918810844421387),
 ('2753184525', 0.7916118502616882),
 ('2284433740', 0.791517972946167),
 ('2560396703', 0.7914338111877441),
 ('2789143569', 0.7912872433662415),
 ('2607822359', 0.7905606031417847),
 ('2591923234', 0.7905330657958984),
 ('2284087212', 0.7901628613471985),
 ('2772126791', 0.7901372909545898),
 ('2292323284', 0.7898640036582947),
 ('800049804', 0.7894905805587769)]

sentences = DatabaseTaggedDocument(dict_cur)
sen = list(sentences)
for row in sen:
    vectors = model.infer_vector(row.words)
    pprint(model.docvecs.most_similar([vectors], topn=20))"""   