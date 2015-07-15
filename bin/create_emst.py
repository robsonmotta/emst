#! /usr/bin/env python
import _mypath
from preprocessing import distancematrix_from_csv

if __name__ == '__main__':
    distance_matrix = distancematrix_from_csv('/Users/robsonmotta/Documents/git/emst/fixtures/calories.csv')
    print distance_matrix
