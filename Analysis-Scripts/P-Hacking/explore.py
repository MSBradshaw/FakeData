import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Data-Uncompressed-Original/CNA.cct', sep='\t')
samples = df.columns
genes = df.iloc[:, 0]
df = df.drop('idx', axis=1)
tdf = df.transpose()
tdf.columns = genes
tdf = tdf.fillna(0)
# do hierarchical clustering on these bad boys
kdf = df.fillna(0)
kmeans = KMeans(n_clusters=3, random_state=0).fit(tdf)

pca = PCA(n_components=2)
pca_res = pca.fit(kdf)

plt_df = pd.DataFrame(
    {'pc1': pca.components_[0], 'pc2': pca.components_[1], 'group': kmeans.labels_, 'sample': samples[1:]})

colors = ['red', 'blue', 'green']

for i, g in enumerate(plt_df['group'].unique()):
    sub = plt_df[plt_df['group'] == g]
    plt.scatter(sub['pc1'], sub['pc2'], color=colors[i])
plt.xlabel('PC1: ' + str(pca.explained_variance_[0])[0:4])
plt.ylabel('PC2: ' + str(pca.explained_variance_[1])[0:4])
plt.show()

# save these groups
save_df = plt_df[['sample','group']]
save_df['group'] = [ 1 if x == 1 else 0 for x in save_df['group']]
save_df.to_csv('Data/P-Hacking/labels.tsv', sep='\t', index=False)

