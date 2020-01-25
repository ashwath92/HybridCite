""" Removed duplicate references """

import pandas as pd 
df = pd.read_csv('/home/ashwath/Programs/ArxivCS/arxiv_testset_contexts.tsv', sep='\t')
# dup_df = df[df.duplicated()]
dedup_df = df[~df.duplicated()]
# Remove the header this time as well.
dedup_df.to_csv('/home/ashwath/Programs/ArxivCS/arxiv_testset_contexts.tsv', sep='\t', header=False)