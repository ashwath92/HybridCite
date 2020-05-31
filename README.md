# HybridCite
*Hybrid Cite Hybrid Citation Recommender Code*

There are separate sections (directories) of code present in this repository for each of the algorithms:
1. **Hyperdoc2vec**: programs to train a Hyperdoc2vec model
2. **BM25**: Indexing programs and Solr configsets
3. **LDA**: Programs to create an LDA model
4. **Paper2vec**: Programs to create Doc2vec and Paper2vec models

The process to create the training and test files are in the CreateTrainingFiles and CreateTestFiles directories respectively.
The online and offline evaluation programs are present under the Evaluation directory along with the resulting data and graphs. The offline evaluation is divided into sub-folders corresponding to each of the data sets. These sub-folders contain the programs, generated data and Tableau workbooks corresponding to the data set. The online evaluation is present as a Flask app.

The running recommender system can be set up by running the code under the HybridCite directory: this is a Flask app too.

Other intermediate programs, databases etc. are present in the various directories.

The data sets created can be downloaded from the following links:

* [ACL-MAG training data](http:/people.aifb.kit.edu/mfa/hybridcite2020/ACLMAG/acl_training_data.txt.gz)
* [arXiv-MAG hd2v training](http:/people.aifb.kit.edu/mfa/hybridcite2020/ArxivMAG/arxiv_hd2v_training.txt.gz)
* [MAG training data](http:/people.aifb.kit.edu/mfa/hybridcite2020/MAG/Training/mag_training_data.txt.gz)
* [MAG50 training data](http:/people.aifb.kit.edu/mfa/hybridcite2020/MAG/Training/mag_training_data_50citationsmin.txt.gz)
* [MAG-Cited training data](http:/people.aifb.kit.edu/mfa/hybridcite2020/MAG/Training/mag_training_data_cited_contexts.txt.gz)
* [MAG-unpaywall-training](http:/people.aifb.kit.edu/mfa/hybridcite2020/UnpaywallMAG/Training/training_no201829109.txt.gz)

## Citing
Please cite our work as follows:
```
Michael Färber, Ashwath Sampath. "HybridCite: A Hybrid Model for Context-Aware Citation Recommendation". Proceedings of the 20th ACM/IEEE Joint Conference on Digital Libraries (JCDL'20), Xi'an, China, 2020.
```
Our paper is available online at https://arxiv.org/pdf/2002.06406.pdf.
