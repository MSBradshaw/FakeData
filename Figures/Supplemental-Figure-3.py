import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import DigitPreferences as dig

res = pd.read_csv('Data/Distribution-Data-Set/CNA-100/test_cna_distribution1.csv')
ran = pd.read_csv('Data/Random-Data-Set/CNA-100/test_cna_random1.csv')
imp = pd.read_csv('Data/Imputation-Data-Set/cna-50/CNA-imputaion-test-1.csv')

real = res[res['labels'] == 'real']

ran = ran[ran['labels'] == 'phony']
ran['labels'] = 'random'

imp.columns = ran.columns
imp = imp[imp['labels'] == 'phony']
imp['labels'] = 'imputed'


res = res[res['labels'] == 'phony']
res['labels'] = 'resampled'

df = pd.concat([real,ran,res,imp])
df = df.reset_index()
dg_df = dig.digit_preference_first_after_dec(df)

sub = dg_df[dg_df['labels'] == 'real']
plt.boxplot(sub.iloc[:,0:10])
plt.xticks([1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9])
plt.xlabel('Digit')
plt.ylabel('Normalized frequency')
plt.savefig('Figures/Supplemental-Figure-3.png')
plt.savefig('Figures/Supplemental-Figure-3.tiff')
plt.show()

quit()

xs = []
ys = []
labels = []
for i in range(10):
    for j,z in enumerate(list(dg_df.iloc[:,i])):
        ys.append(z)
        xs.append(i)
        labels.append(str(dg_df.iloc[j,-1]))


fig, ax = plt.subplots()
sns.boxplot(xs, ys, hue=labels, hue_order=['real', 'random', 'resampled', 'imputed'], ax=ax)
for i, artist in enumerate(ax.artists):
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    for j in range(i * 6, i * 6 + 6):
        line = ax.lines[j]
        # line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

plt.xlabel('Digit')
plt.ylabel('Normalized frequency')
plt.savefig('Figures/Supplemental-Figure-3a.png',dpi=300)
plt.savefig('Figures/Supplemental-Figure-3a.tiff',dpi=300)
plt.show()



