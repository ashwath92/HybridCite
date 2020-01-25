# hybridcite2020
*Hybrid Cite Hybrid Citation Recommender Code*

There are separate sections (directories) of code present in this repository for each of the algorithms:
1. Hyperdoc2vec: programs to train a Hyperdoc2vec model
2. BM25: Indexing programs and Solr configsets
3. LDA: Programs to create an LDA model
4. Paper2vec: Programs to create Doc2vec and Paper2vec models

The process to create the training and test files are in the CreateTrainingFiles and CreateTestFiles directories respectively.
The online and offline evaluation programs are present under the Evaluation directory. The offline evaluation is present as a Flask app.

The running recommender system can be set up by running the code under the HybridCite directory: this is a Flask app too.

Other intermediate programs, databases etc. are present in the various directories.
