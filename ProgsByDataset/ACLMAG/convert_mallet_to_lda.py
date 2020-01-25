import gensim
from gensim.models.wrappers import LdaMallet

ldamallet = LdaMallet.load('/home/ashwath/Programs/ACLAAn/LDA/lda_model.model')
lda = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet, gamma_threshold=0.001, iterations=50)
lda.save('ldanormal_acl.model')