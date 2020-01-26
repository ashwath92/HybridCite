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

The data sets created can be downloaded from the following links:

http:/people.aifb.kit.edu/mfa/hybridcite2020/ACLMAG/acl_training_data.txt.gz
http:/people.aifb.kit.edu/mfa/hybridcite2020/ArxivMAG/arxiv_hd2v_training.txt.gz
http:/people.aifb.kit.edu/mfa/hybridcite2020/MAG/Training/mag_training_data.txt.gz
http:/people.aifb.kit.edu/mfa/hybridcite2020/MAG/Training/mag_training_data_50citationsmin.txt.gz
http:/people.aifb.kit.edu/mfa/hybridcite2020/MAG/Training/mag_training_data_cited_contexts.txt.gz
http:/people.aifb.kit.edu/mfa/hybridcite2020/UnpaywallMAG/Training/training_no201829109.txt.gz

