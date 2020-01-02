import pandas as pd
import time
import numpy as np
import re
import sys

sys.path.insert(0, '/fslhome/mbrad94/Holden/')
import Classification_Utils as cu
import DigitPreferences as dig

def clean_data(df):
    names = df.columns.values
    names[-1] = 'labels'
    df.columns = names
    df = df.replace('nan',0)
    df = df.fillna(0)
    return(df)
#python Classification-Optimized-General-Use.py ../../Data/Imputation-Data-Set/CNA-imputation-train.csv ../../Data/Imputation-Data-Set/CNA-imputation-test.csv imputation-cna-results.csv imputation
#../../Data/Imputation-Data-Set/CNA-imputation-train.csv
#'../../Data/Distribution-Data-Set/test_transcriptomics_distribution.csv'
df = pd.read_csv(sys.argv[1])
df_test = pd.read_csv(sys.argv[2])
#df = pd.read_csv('../../Data/Imputation-Data-Set/cna-50/CNA-imputation-train-6.csv')
#df_test = pd.read_csv('../../Data/Imputation-Data-Set/cna-50/CNA-imputaion-test-6.csv')
print(1)

df = clean_data(df)
df_test = clean_data(df_test)

print(2)
labels = df['labels']
labels_test = df_test['labels']
print(3)
first_df = dig.digit_preference_first_after_dec(df)
second_df = dig.digit_preference_second_after_dec(df)
print(4)
first_df_test = dig.digit_preference_first_after_dec(df_test)
second_df_test = dig.digit_preference_second_after_dec(df_test)
print(5)
df = pd.merge(first_df, second_df, left_index=True, right_index=True)
df_test = pd.merge(first_df_test, second_df_test, left_index=True, right_index=True)
print(6)
NUM_SPLITS = 10 # number of train/test splits in cross validation

#drop columns not intended for training
df = df.drop(['sample_id_x', 'sample_id_y', 'labels_x', 'labels_y' ], axis=1)
df_test = df_test.drop(['sample_id_x', 'sample_id_y', 'labels_x', 'labels_y' ], axis=1)

print('KNN')
start = time.time()
knn = cu.knn_model_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start)/60, "minutes")

print('SVC')
start = time.time()
svc = cu.SVC_model_crossval(df, labels, NUM_SPLITS)
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

print('LR')
#start = time.time()
#lr = cu.logistic_regression_model_crossval(df, labels, NUM_SPLITS)
#end = time.time()
#print("Runtime:", (end - start)/60, "minutes")

print('MLP')
start = time.time()
mlp = cu.mlp_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start)/60, "minutes")

### This is commented out so that you do not call predictinos until you are done finalizing the training sets!!!
### DO NOT RUN MORE THAN ONCE! THAT IS CHEATING MYREE!
#lr_pred = lr.predict(df_test)
#lr_result = lr.score(df_test, labels_test)

rf_pred = rf.predict(df_test)
rf_result = rf.score(df_test, labels_test)

svc_pred = svc.predict(df_test)
svc_result = svc.score(df_test, labels_test)

gbc_pred = gbc.predict(df_test)
gbc_result = gbc.score(df_test, labels_test)

gnb_pred = gnb.predict(df_test)
gnb_result = gnb.score(df_test, labels_test)

knn_pred = knn.predict(df_test)
knn_result = knn.score(df_test, labels_test)

mlp_pred = mlp.predict(df_test)
mlp_result = mlp.score(df_test, labels_test)


print(rf_result)
print(svc_result)
print(gbc_result)
print(gnb_result)
print(knn_result)
print(mlp_result)
results = [rf_result,svc_result,gbc_result,gnb_result,knn_result,mlp_result]
learners = ['Random Forest','SVC','GBC','Naive Bayes','KNN','MLP']
t='imputation'
t = sys.argv[4]

type = [t,t,t,t,t,t]
final = pd.DataFrame({'score':results,'learner':learners,'type':type})
final.head()
final.to_csv(sys.argv[3])
