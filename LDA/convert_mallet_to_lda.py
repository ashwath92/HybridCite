import gensim
from gensim.models.wrappers import LdaMallet
# If mallet doesn't work, use normal LDA.
from gensim.models.ldamodel import LdaModel
ldamallet = LdaMallet.load('/home/ashwath/Programs/MAGCS/LDA/ldamallet_mag50.model')
lda = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet, gamma_threshold=0.001, iterations=50)
lda.save('lda_mag50.model')