import pandas as pd
import time
import numpy as np
import re
import sys
import random


# path to root of this project so Classification_Utils and DigitPreferences can be imported
#sys.path.insert(0, '/Users/michael/Holden/')
# Fiji path
sys.path.insert(0, '/Users/mibr6115/Holden/')
import Classification_Utils as cu
import DigitPreferences as dig

# USE OF THIS FILE This script trainings and tests 6 machine learning models on digit frequency data. Test and train
# sets are read in, have their features random down sampled to be N in total and converted to 20 features
# representing the frequency of digits 0-9 in the first and second place after the decimal. Models are then trained
# and tested, model accuracies are then output to a .csv log file.

# ARGUMENTS
# 1 TRAINING DATA CSV
# 2 TESTING DATA CSV
# 3 NAME OF RESULT CSV LOG TO WRITE TO
# 4 TYPE OF DATA BEING USED (random, resampled, imputed)


def clean_data(df):
    names = df.columns.values
    names[-1] = 'labels'
    df.columns = names
    df = df.replace('nan', 0)
    df = df.fillna(0)
    return (df)

# TODO write a function to down sample the number of features, default all, add a command line parameter for number
#  of features to use

# Takes training data, test data and n (number of feature to select). Randomly selected n features without replacement
# subsets the original data frames with the randomly selected portions and returns them
def stachastic_downsample(train, test, n=-1):
    print(train.shape)
    if n == -1:
        # assuming incoming data is a np array
        number_of_features = train.shape[1]
    # choose n number from 0 - number of features in data. without replacement
    cols = random.sample(range(train.shape[1]),n)
    return train.iloc[:,cols], test.iloc[:,cols]


# python Classification-Optimized-General-Use.py ../../Data/Imputation-Data-Set/CNA-imputation-train.csv ../../Data/Imputation-Data-Set/CNA-imputation-test.csv imputation-cna-results.csv imputation
# ../../Data/Imputation-Data-Set/CNA-imputation-train.csv
# '../../Data/Distribution-Data-Set/test_transcriptomics_distribution.csv'
df = pd.read_csv(sys.argv[1])
df_test = pd.read_csv(sys.argv[2])
number_of_features = int(sys.argv[5])


# df = pd.read_csv('../../Data/Imputation-Data-Set/cna-50/CNA-imputation-train-6.csv')
# df_test = pd.read_csv('../../Data/Imputation-Data-Set/cna-50/CNA-imputaion-test-6.csv')
print(1)

df = clean_data(df)
df_test = clean_data(df_test)

print(2)
labels = df['labels']
labels_test = df_test['labels']

# remove the labels column and re-add it
df = df.drop('labels',axis=1)
df_test = df_test.drop('labels',axis=1)

df, df_test = stachastic_downsample(df,df_test,number_of_features)

df['labels'] = labels
df_test['labels'] = labels_test


print('Size: ', str(df.shape))

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
NUM_SPLITS = 10  # number of train/test splits in cross validation

# drop columns not intended for training
df = df.drop(['sample_id_x', 'sample_id_y', 'labels_x', 'labels_y'], axis=1)
df_test = df_test.drop(['sample_id_x', 'sample_id_y', 'labels_x', 'labels_y'], axis=1)

print('KNN')
start = time.time()
knn = cu.knn_model_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start) / 60, "minutes")

# print('SVC')
# start = time.time()
# svc = cu.SVC_model_crossval(df, labels, NUM_SPLITS)
# end = time.time()
# print("Runtime:", (end - start) / 60, "minutes")

print('RF')
start = time.time()
rf = cu.randomforest_model_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start) / 60, "minutes")

print('Gradient Boosting')
start = time.time()
gbc = cu.gradient_boosting_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start) / 60, "minutes")

print('Niave Bayes')
start = time.time()
gnb = cu.bayes_gaussian_model_crossval(df, labels, NUM_SPLITS)
end = time.time()
print("Runtime:", (end - start) / 60, "minutes")

print('LR')
# start = time.time()
# lr = cu.logistic_regression_model_crossval(df, labels, NUM_SPLITS)
# end = time.time()
# print("Runtime:", (end - start)/60, "minutes")

# print('MLP')
# start = time.time()
# mlp = cu.mlp_crossval(df, labels, NUM_SPLITS)
# end = time.time()
# print("Runtime:", (end - start) / 60, "minutes")
#

### This is commented out so that you do not call predictinos until you are done finalizing the training sets!!!
### DO NOT RUN MORE THAN ONCE! THAT IS CHEATING MYREE!
# lr_pred = lr.predict(df_test)
# lr_result = lr.score(df_test, labels_test)

rf_pred = rf.predict(df_test)
rf_result = rf.score(df_test, labels_test)

# svc_pred = svc.predict(df_test)
# svc_result = svc.score(df_test, labels_test)

gbc_pred = gbc.predict(df_test)
gbc_result = gbc.score(df_test, labels_test)

gnb_pred = gnb.predict(df_test)
gnb_result = gnb.score(df_test, labels_test)

knn_pred = knn.predict(df_test)
knn_result = knn.score(df_test, labels_test)
#
# mlp_pred = mlp.predict(df_test)
# mlp_result = mlp.score(df_test, labels_test)
#
print(rf_result)
# print(svc_result)
print(gbc_result)
print(gnb_result)
print(knn_result)
# print(mlp_result)
results = [rf_result, gbc_result, gnb_result, knn_result]
learners = ['Random Forest', 'GBC', 'Naive Bayes', 'KNN']
t = 'imputation'
t = sys.argv[4]
numbers = [number_of_features,number_of_features,number_of_features,number_of_features]
type = [t, t, t, t]
final = pd.DataFrame({'score': results, 'learner': learners, 'type': type, 'features': numbers})
final.head()
#final.to_csv(sys.argv[3])
with open(sys.argv[3], 'a') as f:
    print('Saving to file ', sys.argv[3])	
    final.to_csv(f, header=False)
