import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import random
import sys

def resample_data(real_data, n):
    """
    sample uniformly at random for the set of real values
    :param real_data: list of values
    :param n: number of fake values to create
    :return: list of n values sampled from real data
    """
    print(real_data)
    print(type(real_data))
    items = []
    for i in range(n):
        items.append(random.sample(real_data, 1)[0])
    return items

random.seed(int(sys.argv[1]))

df = pd.read_csv('Data/Data-Uncompressed-Original/CNA.cct', sep='\t')
samples = df.columns
genes = df.iloc[:, 0]
df = df.drop('idx', axis=1)
tdf = df.transpose()
tdf.columns = genes
tdf = tdf.fillna(0)

label_df = pd.read_csv('Data/P-Hacking/labels.tsv', sep='\t')

tdf['label'] = list(label_df['group'])

smallest = 1
smallest_name = None
count_not_sig = 0
non_sig_genes = []
# p-value for each gene across the groups
for col in tdf.columns:
    g0 = tdf[tdf['label'] == 0][col]
    g1 = tdf[tdf['label'] == 1][col]
    pv = stats.ttest_ind(g0, g1).pvalue
    if pv < smallest:
        smallest = pv
        smallest_name = col
    if pv > 0.05 / tdf.shape[1]:
        count_not_sig += 1
        non_sig_genes.append(col)

small_df = tdf[tdf['label'] == 0]
big_df = tdf[tdf['label'] == 1]

small_df = small_df.reset_index()

fake_data = small_df['index']
fake_data.columns = ['index']
still_not_sig = []
for gene in tdf.columns:
    if gene in non_sig_genes:
        # if the control group's mean is larger, use the lower value set of treatment values
        if big_df[gene].mean() > small_df[gene].mean():
            keep_small_df = small_df[small_df[gene] < small_df[gene].mean()]
            augment_small_df = small_df[small_df[gene] >= small_df[gene].mean()]
        # if the control group's mean is smaller, use the higher value set of treatment values
        else:
            keep_small_df = small_df[small_df[gene] > small_df[gene].mean()]
            augment_small_df = small_df[small_df[gene] <= small_df[gene].mean()]

        new_vals = resample_data(list(keep_small_df[gene]), augment_small_df.shape[0])
        fake_treatment_group = new_vals + list(keep_small_df[gene])
        control_group = big_df[gene]
        pval = stats.ttest_ind(fake_treatment_group, control_group).pvalue
        if pval > 0.05 / tdf.shape[1]:
            still_not_sig.append(genes)

        temp_df = pd.DataFrame({'index': list(keep_small_df['index']) + list(augment_small_df['index']),
                                gene: list(keep_small_df[gene]) + new_vals})
        fake_data = pd.merge(fake_data, temp_df, on='index')
    else:
        fake_data = pd.merge(small_df[['index', gene]], fake_data, on='index')

df = pd.concat([fake_data, big_df])

df.to_csv('Data/P-Hacking/Resampled/resampled_' + sys.argv[1] + '.csv', index=False)