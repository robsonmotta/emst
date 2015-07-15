#! /usr/bin/env python
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cross_validation import StratifiedKFold

def preprocessed_dataset_from_csv(csv_filename, do_standardization=True, do_normalization=False):
    data_frame = pd.DataFrame.from_csv(csv_filename, index_col=None)
    X = data_frame.values[:,:-1]
    y = data_frame.values[:,-1]

    X_preprocessed = X
    if do_standardization:    
        std_scale = preprocessing.StandardScaler().fit(X)
        X_preprocessed = std_scale.transform(X)
    elif do_normalization:
        norm_scale = preprocessing.MinMaxScaler().fit(X)
        X_preprocessed = norm_scale.transform(X)

    return X_preprocessed, y

def split_folds(X, y, k_fold=10):
    folds = StratifiedKFold(y, k_fold)
    X_train_10fold, X_test_10fold, y_train_10fold, y_test_10fold = [], [], [], []
    for train_index, test_index in folds:
        X_train_10fold.append(X[train_index])
        X_test_10fold.append(X[test_index])
        y_train_10fold.append(y[train_index])
        y_test_10fold.append(y[test_index])
    return X_train_10fold, X_test_10fold, y_train_10fold, y_test_10fold

def calculate_distance_matrix(points):
    return euclidean_distances(points)

def splitted_data_from_csv(csv_filename, do_standardization=True, do_normalization=False, k_fold=10):
    # read and preprocessing
    X, y = preprocessed_dataset_from_csv(csv_filename, do_standardization, do_normalization)

    # split 10fold cross-validation
    X_train_10fold, X_test_10fold, y_train_10fold, y_test_10fold = split_folds(X, y, k_fold=k_fold)

    # calculate distance matrix
    distance_matrix_10fold = [calculate_distance_matrix(X_train_item) for X_train_item in X_train_10fold]

    return (X, y), (X_train_10fold, X_test_10fold, y_train_10fold, y_test_10fold), distance_matrix_10fold
