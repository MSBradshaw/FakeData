import pandas as pd
import time
import numpy as np
import re
import sys
import random

# path to root of this project so Classification_Utils and DigitPreferences can be imported
# sys.path.insert(0, '/Users/michael/Holden/')
# Fiji path
sys.path.insert(0, '/Users/mibr6115/Holden/')
import Classification_Utils as cu
import DigitPreferences as dig


# USE OF THIS FILE This script trainings and tests 6 machine learning models on digit frequency data. Test and train
# sets are read in, have their features random down sampled to be N in total and converted to 20 features
# representing the frequency of digits 0-9 in the first and second place after the decimal. Models are then trained
# and tested, model accuracies are then output to a .csv log file.

# ARGUMENTS
# 1 TRAINING DATA number
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


# python Classification-Optimized-General-Use.py ../../Data/Imputation-Data-Set/CNA-imputation-train.csv ../../Data/Imputation-Data-Set/CNA-imputation-test.csv imputation-cna-results.csv imputation
# ../../Data/Imputation-Data-Set/CNA-imputation-train.csv
# '../../Data/Distribution-Data-Set/test_transcriptomics_distribution.csv'
df = pd.read_csv(
    '/Users/mibr6115/Holden/Data/Distribution-Data-Set/CNA-100/' + 'train_cna_distribution' + sys.argv[1] + '.csv')

df_test = None
for i in range(1,10,1):
    df_test = pd.read_csv('/Users/mibr6115/Holden/Data/Partially-Fake-Resampled/CNA/' + 'test_partially_resampled' + sys.argv[1] + '_fake_0.' + str(i) + '.csv')
    number_of_features = int(i)
    print('/Users/mibr6115/Holden/Data/Partially-Fake-Resampled/CNA/' + 'test_partially_resampled' + sys.argv[1] + '_fake_0.' + str(i) + '.csv')
    # df = pd.read_csv('../../Data/Imputation-Data-Set/cna-50/CNA-imputation-train-6.csv')
    # df_test = pd.read_csv('../../Data/Imputation-Data-Set/cna-50/CNA-imputaion-test-6.csv')
    print(1)

    df = clean_data(df)
    df_test = clean_data(df_test)

    print(2)
    labels = df['labels']
    labels_test = df_test['labels']

    # remove the labels column and re-add it
    df = df.drop('labels', axis=1)
    df_test = df_test.drop('labels', axis=1)

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
    df2 = pd.merge(first_df, second_df, left_index=True, right_index=True)
    df_test2 = pd.merge(first_df_test, second_df_test, left_index=True, right_index=True)
    print(6)
    NUM_SPLITS = 10  # number of train/test splits in cross validation

    # drop columns not intended for training
    df2 = df2.drop(['sample_id_x', 'sample_id_y', 'labels_x', 'labels_y'], axis=1)
    df_test2 = df_test2.drop(['sample_id_x', 'sample_id_y', 'labels_x', 'labels_y'], axis=1)

    print('KNN')
    start = time.time()
    knn = cu.knn_model_crossval(df2, labels, NUM_SPLITS)
    end = time.time()
    print("Runtime:", (end - start) / 60, "minutes")

    # print('SVC')
    # start = time.time()
    # svc = cu.SVC_model_crossval(df, labels, NUM_SPLITS)
    # end = time.time()
    # print("Runtime:", (end - start) / 60, "minutes")

    print('RF')
    start = time.time()
    rf = cu.randomforest_model_crossval(df2, labels, NUM_SPLITS)
    end = time.time()
    print("Runtime:", (end - start) / 60, "minutes")

    print('Gradient Boosting')
    start = time.time()
    gbc = cu.gradient_boosting_crossval(df2, labels, NUM_SPLITS)
    end = time.time()
    print("Runtime:", (end - start) / 60, "minutes")

    print('Niave Bayes')
    start = time.time()
    gnb = cu.bayes_gaussian_model_crossval(df2, labels, NUM_SPLITS)
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

    rf_pred = rf.predict(df_test2)
    rf_result = rf.score(df_test2, labels_test)

    # svc_pred = svc.predict(df_test)
    # svc_result = svc.score(df_test, labels_test)

    gbc_pred = gbc.predict(df_test2)
    gbc_result = gbc.score(df_test2, labels_test)

    gnb_pred = gnb.predict(df_test2)
    gnb_result = gnb.score(df_test2, labels_test)

    knn_pred = knn.predict(df_test2)
    knn_result = knn.score(df_test2, labels_test)
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
    t = sys.argv[4] + str(i)
    fakeness = '0.' + str(i) + '0'
    numbers = [fakeness, fakeness, fakeness, fakeness]
    type = [t, t, t, t]
    final = pd.DataFrame({'score': results, 'learner': learners, 'type': type, 'fakeness': numbers})
    final.head()
    # final.to_csv(sys.argv[3])
    with open(sys.argv[3], 'a') as f:
        print('Saving to file ', sys.argv[3])
        final.to_csv(f, header=False)
