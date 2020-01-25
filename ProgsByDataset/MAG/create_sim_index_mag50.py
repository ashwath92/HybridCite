from gensim import corpora, models, similarities
from gensim.models.wrappers import LdaMallet
# If mallet doesn't work, use normal LDA.
from gensim.models.ldamodel import LdaModel
import subprocess

corpus = corpora.MmCorpus('/home/ashwath/Programs/MAGCS/LDA/mag50_bow_corpus.mm')
try:
    ldamallet = LdaMallet.load('/home/ashwath/Programs/MAGCS/LDA/ldamallet_mag50.model')
    #vec_bow_test = id2word_dictionary.doc2bow(['test'])
    #vec_ldamallet = ldamallet[vec_bow_test]
except subprocess.CalledProcessError:
    print("LDA MALLET COULDN'T READ INSTANCE FILE. USING NORMAL LDA INSTEAD")
    ldamallet = LdaModel.load('/home/ashwath/Programs/MAGCS/LDA/lda_mag50.model')

index = similarities.MatrixSimilarity(ldamallet[corpus])
index.save("simIndexMag50.index")
