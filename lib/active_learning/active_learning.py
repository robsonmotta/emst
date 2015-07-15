#! /usr/bin/env python

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def classify_folds(X_train_10fold, X_test_10fold, y_train_10fold, y_test_10fold, n_fold=10, n_neighbors=5):
    accuracies = []
    for fold in range(n_fold):
        X_train = X_train_10fold[fold]
        X_test = X_test_10fold[fold]
        y_train = y_train_10fold[fold]
        y_test = y_test_10fold[fold]
        classifier = KNeighborsClassifier(n_neighbors)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(metrics.accuracy_score(y_test, y_pred))
    return np.mean(accuracies)
