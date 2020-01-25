

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
df = pd.read_pickle('/home/ashwath/Programs/MAGCS/Pickles/paperwisemetrics_mag_3models_df.pickle')
df = df.drop(['hd2v_recommendations', 'bm25_recommendations', 'hd2v_binary','bm25_binary', 'ground_truth'], axis=1)
mean_series = df.mean()
mean_series.to_csv('/home/ashwath/Programs/MAGCS/Evaluation/meanmetrics_mag_nolda.tsv', sep='\t', index=True, header=False)
print("C'est fini.")