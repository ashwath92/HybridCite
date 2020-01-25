""" Removed duplicate references """

import pandas as pd 
df = pd.read_csv('/home/ashwath/Programs/ArxivCS/AdditionalOutputs/arxivmag_references.tsv', sep='\t')
# dup_df = df[df.duplicated()].shape is (9927,2)
dedup_df = df[~df.duplicated()]
# Remove the header this time as well.
dedup_df.to_csv('/home/ashwath/Programs/ArxivCS/AdditionalOutputs/arxivmag_references_deduplicated.tsv', sep='\t', header=False)