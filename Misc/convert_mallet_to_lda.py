import gensim
ldamallet = LdaMallet.load('/home/ashwath/Programs/ArxivCS/LDA/ldamallet_arxiv.model')
lda = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet, gamma_threshold=0.001, iterations=50)
lda.save('lda_arxiv.model')