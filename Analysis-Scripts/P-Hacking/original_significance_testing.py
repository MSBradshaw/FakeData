import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats


df = pd.read_csv('Data/Data-Uncompressed-Original/CNA.cct', sep='\t')
samples = df.columns
genes = df.iloc[:, 0]
df = df.drop('idx', axis=1)
tdf = df.transpose()
tdf.columns = genes
tdf = tdf.fillna(0)

label_df = pd.read_csv('Data/P-Hacking/labels.tsv',sep='\t')

tdf['label'] = list(label_df['group'])

smallest = 1
smallest_name = None
count_not_sig = 0
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
        print(col, pv)

# this one is close to significant, but not quite TOR2A p=2.466003395495693e-05
gene = 'TOR2A'

# plot the distribution of values
g0 = tdf[tdf['label'] == 0][gene]
g1 = tdf[tdf['label'] == 1][gene]

plt.boxplot([g0,g1])
plt.show()

# the values all need to be skewed down
# take the samples that have above the mean and make fake data for them based on what was found in the samples below the mean


