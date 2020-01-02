"""
Provides functions to package together common classification steps
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn import tree

#########################
#
# Classifiers
#
#########################

"""
Abstraction of cross-validation score calculation and stat printing
Called from all specific make_x_model functions

Args:
    model: instance of any sklearn classification model
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string): scoring method
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

Returns:
    The given model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def fit_model(model, data, labels, num_splits, scoring):
    cv = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.3, random_state=0)
    scores = cross_val_score(model, data, labels, cv=cv, scoring=scoring)
    #print('Scores:',scores)
    print('%s: %0.2f (+/- %0.2f)' % (scoring, scores.mean(), scores.std() * 2))
    return model.fit(data, labels)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    K Nearest Neighbors classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def knn_model_crossval(data, labels, num_splits, scoring='accuracy'):
    knn = KNeighborsClassifier()
    return fit_model(knn, data, labels, num_splits, scoring)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    Decision Tree classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def decisiontree_model_crossval(data, labels, num_splits, scoring='accuracy'):
    dt = tree.DecisionTreeClassifier(random_state=0)
    return fit_model(dt, data, labels, num_splits, scoring)


"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    Random forest classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def randomforest_model_crossval(data, labels, num_splits, scoring='accuracy'):
    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    return fit_model(rf, data, labels, num_splits, scoring)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    Naive Bayes Multinomial classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def bayes_multinomial_model_crossval(data, labels, num_splits, scoring='accuracy'):
    mnb = MultinomialNB()
    return fit_model(mnb, data, labels, num_splits, scoring)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    Naive Bayes Gaussian classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def bayes_gaussian_model_crossval(data, labels, num_splits, scoring='accuracy'):
    gnb = GaussianNB()
    return fit_model(gnb, data, labels, num_splits, scoring)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    Logistic Regression classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def logistic_regression_model_crossval(data, labels, num_splits, scoring='accuracy'):
    lr = LogisticRegression(random_state=0)
    return fit_model(lr, data, labels, num_splits, scoring)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    SVC model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def SVC_model_crossval(data, labels, num_splits, scoring='accuracy'):
    svc = SVC(kernel='linear', probability=True, random_state=0)
    return fit_model(svc, data, labels, num_splits, scoring)


"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    List of SVC classification models with various kernels fitted on all inputted data
    Prints mean cross-validation scores and 95% confidence intervals
"""
def SVC_models_crossval(data, labels, num_splits, scoring='accuracy'):
    C = 1.0  # SVM regularization parameter
    models = (SVC(kernel='linear', C=C, probability=True, random_state=0),
              LinearSVC(C=C, random_state=0),
              SVC(kernel='rbf', gamma=0.7, C=C, probability=True, random_state=0),
              SVC(kernel='poly', degree=3, C=C, probability=True, random_state=0))

    # Fit all the models
    models = (fit_model(clf, data, labels, num_splits, scoring) for clf in models)
    model_list = list(models)

    return model_list

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    Gradient Boosting Classifier model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def gradient_boosting_crossval(data, labels, num_splits, scoring='accuracy'):
    gbc = GradientBoostingClassifier(random_state=0)
    return fit_model(gbc, data, labels, num_splits, scoring)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    scoring (string, optional): scoring method. Defaults to accuracy score

Returns:
    Multi Layer Perceptron model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def mlp_crossval(data, labels, num_splits, scoring='accuracy'):
    mlp = MLPClassifier(hidden_layer_sizes=(30), random_state=0)
    return fit_model(mlp, data, labels, num_splits, scoring)

"""
Use a classification model to make label predictions on test data set

Args:
    model: classification model
    data (dataframe): new data to be labelled by the model
    labels (list of strings): list of correct labels for the input data
    print_details (boolean, optional): Determines whether to print prediction information. Defaults to True

Returns:
    List of strings: List of predicted labels
    Prints accuracy score, as well as the predicted and actual labels
"""
def make_test_prediction(model, data, labels, print_details=True):
    pred = model.predict(data)
    if(print_details):
        print('score', accuracy_score(pred, labels))
        print('pred', pred)
        print('actual', labels)

    return pred

"""
Args:
    model: SKLearn classification model
    data (dataframe): test data
    idx (int): index of sample to show prediction probabilities for

Returns:
    Prints each class's prediction probability for the specified data sample
"""
def show_prediction_probabilities(model, data, idx):
    pred_probabilities = model.predict_proba(data)
    classes = model.classes_

    print('Prediction probabilities for sample:')
    for prob in zip(classes, pred_probabilities[idx]):
        print(prob[0], ':', prob[1])


#########################
#
# Dataframe adjustments
#
#########################

""" Reads in tab separated files containing a 'Peptide' column

Args:
    file_dir (string): path to directory containing files
    file_paths (list of strings): list of file names of csvs to read

Returns:
    dataframe containing data from all csvs referenced by file_paths. Dataframe index is 'Peptide'; each column represents a single sample.
"""
def combine_csvs(file_dir, file_names):

    dfs = []

    for file in file_names:
        df = pd.read_csv(file_dir + file, sep='\t', lineterminator='\r')
        dfs.append(df)

    combined_df = pd.DataFrame()
    for df in dfs:
        df.set_index('Peptide', inplace=True)
        combined_df = combined_df.join(df, how='outer')

    return combined_df


"""
Rename columns so that all instances of "before" are replaced with "after"

Example usage:
my_df.columns = rename_columns(my_df, 'Adult', 'Human')

Args:
    df (dataframe)
    before (string)
    after (string)

Returns:
    List of strings: a list of the new column names
"""
def rename_columns(df, before, after):
    columns = df.columns.values.tolist()
    new_columns = []
    for column in columns:
        new_column = re.sub(before, after, column)
        new_columns.append(new_column)

    return new_columns


"""
Args:
    columns (list of strings): list of all column names in df
    organ_to_columns (dict): mapping of each organ to its column names {str: list of str}

Returns:
    List of strings representing the labels for each dataframe column
"""
def get_labels(columns, organ_to_columns):
    labels = []

    for column in columns:
        key = next(key for key, value in organ_to_columns.items() if column in value)
        labels.append(key)

    return labels

"""
Args:
    df (dataframe)
    min_samples (int)
    min_tissues (int)
    max_tissues (int)
    tissues (list of strings)
    imputed_val (int): impute value for non-observed peptides in df

Returns:
    df filtered to only contain peptides present in at least min_samples samples of a single tissue, for a number of tissues specified by min_tissues and max_tissues
"""
def filter_peptides_by_samples_and_tissues(df, min_samples, min_tissues, max_tissues, tissues, imputed_val):
    df_cols = df.columns.values.tolist()
    organ_counts = {}

    for tissue in tissues:
        cols = [col for col in df_cols if col.startswith(tissue)] # Get corresponding list of column names
        organ_counts[tissue] = (df[cols] != imputed_val).sum(1) # count number of samples with non-imputed abundance for each protein

    tallys = 1 * (organ_counts[tissues[0]] >= min_samples)
    for t in tissues[1:]:
        tallys += 1 * (organ_counts[t] >= min_samples)

    new_df = df[(tallys >= min_tissues) & (tallys <= max_tissues)]
    return new_df

"""
Args:
    df (dataframe): rows are peptide/proteins, columns are samples, data = abundance values

Returns:
    dataframe transformed so that rows represent all pairwise peptide/protein ratios
"""
def pairwise_transform(df):

    index = df.index.values.tolist()

    new_indices = []
    new_data = {}

    for col in df.columns:                             # For each sample
        for i in index:                                # For each pair of peptides
            for j in index:
                ratio = df.loc[i, col]/df.loc[j, col]  # Calculate ratio
                new_index = i + '/' + j                # Create new index value 'i/j'
                if new_index not in new_indices:
                    new_indices.append(new_index)      # Add new index to list

                data = new_data.get(col, list())       # Add ratio to corresponding data
                data.append(ratio)

                new_data[col] = data

    transformed_df = pd.DataFrame(new_data, columns=df.columns, index=new_indices)
    return transformed_df


"""
Fits new data to training features so that it can be classified

Args:
    original_df (dataframe): data used to train classification model
    new_df (dataframe): new data to be classified
    features_to_keep(list of strings, optional): list of selected features kept in training data

Returns:
    dataframe: new_df joined to the features of the training data. This dataframe can now be classified by a model trained with original_df
"""
def fit_new_data(original_df, new_df, features_to_keep=None):

    fitted_data = original_df.join(new_df)

    fitted_data.iloc[:,:] = np.log2(fitted_data.iloc[:,:])
    fitted_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    fitted_data = fitted_data.fillna(fitted_data.min().min()/2)

    median_of_medians = fitted_data.median().median()
    fitted_data /= fitted_data.median(axis=0) # divide each value by sample median
    fitted_data *= median_of_medians

    fitted_data.drop(original_df.columns, axis=1, inplace=True)

    fitted_data = fitted_data.T

    if(features_to_keep is not None):
        fitted_data = fitted_data[features_to_keep]

    return fitted_data

"""
Transforms dataframe values to 0 or 1 to represent presence/absence
"""
def abundance_to_binary(df):
    mode = df.mode().iloc[0,0]
    df = df.applymap(lambda x: 0 if x==mode else 1)
    return df

#########################
#
# Plotting
#
#########################

""" Creates a mapping of each tissue to all corresponding columns in the dataframe. Assumes columns contain the names of the tissues (e.g. column names might look like 'Lung_01', 'Lung_02', 'Brain_01' etc.

Args:
    df (dataframe): columns represent samples, named with the tissue type
    list_of_tissues (list of strings): all tissues represented in the dataframe

Returns:
    dict {string: list of strings} where keys are tissues and values are corresponding column names
"""
def map_tissues_to_columns(df, list_of_tissues):

    tissues_to_columns = dict([(key, []) for key in list_of_tissues])

    for column_name in df.columns.values.tolist():
        for tissue in list_of_tissues:
            if tissue in column_name:
                tissues_to_columns[tissue].append(column_name)
                continue

    return tissues_to_columns

"""
From SKLearn ConfusionMatrix documentation:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""
Modified from SKLearn ConfusionMatrix documentation:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

Args:
    y_test (list of strings): actual labels
    y_pred (list of strings): predicted labels
    groups (list of strings): list of all unique labels
    title(string, optional): chart title. Defaults to 'Confusion Matrix'
"""
def show_confusion_matrices(y_test, y_pred, groups, title='Confusion Matrix'):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=groups,
                          title= title + ', Without Normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=groups, normalize=True,
                          title= 'Normalized ' + title)

    plt.show()


"""
Args:
    df (dataframe)
    labels (list of strings): List of column labels for each column in df

Returns:
    dict({string: list of strings}): key is the name of an organ/tissue, value is a sorted list of the top proteins expressed in that organ/tissue by mean abundance
"""
def get_descending_abundances(df, labels):
    labelled_df = df
    labelled_df.columns = labels

    label_to_proteins = {} # {label: list of proteins}
    for label in labels:
        sub_df = labelled_df[label]
        sorted_proteins = sub_df.mean(axis=1).sort_values(ascending=False)
        label_to_proteins[label] = sorted_proteins.index.values

    return label_to_proteins

"""
Args:
    labels_to_proteins (dict {string : list of strings}): output from get_descending_abundances
    label (string): organ/tissue
    n (int): number of proteins to get

Returns:
    list of strings: top n proteins by abundance for the given organ/tissue
"""
def n_most_abundant(labels_to_proteins, label, n):

    top_proteins = labels_to_proteins[label][:n]
    return top_proteins


#########################
#
# Top distinguishing features
#
#########################

"""Transforms a dataframe to keep only the k rows most significant in terms of group-wise ANOVA-F value

Args:
    df (dataframe): rows are proteins/peptides, columns are samples
    labels (list of strings): list of corresponding labels for df columns
    k (int): number of features to keep

Returns:
    transformed df with only the k best features kept
"""
def keep_k_best_features(df, labels, k):

    select_k_best_classifier = SelectKBest(k=k)
    kbest = select_k_best_classifier.fit_transform(df[:].T, labels)

    fit_transformed_features = select_k_best_classifier.get_support()

    kbest_df = pd.DataFrame(df, index = df.T.columns[fit_transformed_features])
    return kbest_df

"""Transforms a dataframe to keep only the top k percentile rows most significant in terms of group-wise ANOVA-F value

Args:
    df (dataframe): rows are proteins/peptides, columns are samples
    labels (list of strings): list of corresponding labels for df columns
    k (int): percentile of features to keep

Returns:
    transformed df with only the k percentile best features kept
"""
def keep_percentile_features(df, labels, k):

    select_k_percentile_classifier = SelectPercentile(percentile=k)
    kbest = select_k_percentile_classifier.fit_transform(df[:].T, labels)

    fit_transformed_features = select_k_percentile_classifier.get_support()

    kbest_df = pd.DataFrame(df, index = df.T.columns[fit_transformed_features])
    return kbest_df
