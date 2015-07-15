#! /usr/bin/env python
import _mypath
from preprocessing import splitted_data_from_csv
from active_learning import classify_folds

if __name__ == '__main__':
    dataset, dataset_folds, distance_matrix = splitted_data_from_csv('/Users/robsonmotta/Documents/git/emst/fixtures/iris.csv')
    dataset_X, dataset_y = dataset
    dataset_X_train_10fold, dataset_X_test_10fold, dataset_y_train_10fold, dataset_y_test_10fold = dataset_folds

    accuracy_mean = classify_folds(dataset_X_train_10fold, dataset_X_test_10fold, dataset_y_train_10fold, dataset_y_test_10fold)
    
    print "accuracy_mean", accuracy_mean
