import pandas as pd
import time
import numpy as np
import re
import sys

sys.path.insert(0, '/fslhome/mbrad94/Holden/')
import Classification_Utils as cu

print(1)
df = pd.read_csv(sys.argv[1])
df_test = pd.read_csv(sys.argv[2])

print(2)
labels = df['labels']
df = df.drop(columns=['labels'])
labels_test = df_test['labels']
df_test = df_test.drop(columns=['labels'])
print(3)

# impute the NA with 0
df = df.fillna(0)
df_test = df_test.fillna(0)

df.head()
df_test.head()

NUM_SPLITS = 10 # number of train/test splits in cross validation

print('KNN')
start = time.time()
knn = cu.knn_model_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start)/60, "minutes")

print('RF')
start = time.time()
rf = cu.randomforest_model_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start)/60, "minutes")

print('Gradient Boosting')
start = time.time()
gbc = cu.gradient_boosting_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start)/60, "minutes")

print('Niave Bayes')
start = time.time()
gnb = cu.bayes_gaussian_model_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start)/60, "minutes")

### This is commented out so that you do not call predictinos until you are done finalizing the training sets!!!
### DO NOT RUN MORE THAN ONCE! THAT IS CHEATING MYREE!
#lr_pred = lr.predict(df_test)
#lr_result = lr.score(df_test, labels_test)

rf_pred = rf.predict(df_test)
rf_result = rf.score(df_test, labels_test)

gbc_pred = gbc.predict(df_test)
gbc_result = gbc.score(df_test, labels_test)

gnb_pred = gnb.predict(df_test)
gnb_result = gnb.score(df_test, labels_test)

knn_pred = knn.predict(df_test)
knn_result = knn.score(df_test, labels_test)

#print(lr_result)
#print(mnb_result)
print(rf_result)
print(gbc_result)
print(gnb_result)
print(knn_result)
results = [rf_result,gbc_result,gnb_result,knn_result]
learners = ['Random Forest','GBC','Naive Bayes','KNN']
final = pd.DataFrame({'score':results,'learner':learners})
final.head()
final.to_csv(sys.argv[3])
#plot = ggplot(aes(x='learner', weight='score'), data=final) + geom_bar() + ggtitle('Resampling Test Transcriptomics Accuracy') + xlab('Model') + ylab('Accuracy')
#plot.save(sys.argv[3])
