#!/usr/bin/python
# 
# Predicting Bitcoin transactions
#
# Author: Chaney Lin
# Date: March 2015
#
#

import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv
import time

from utils.svd import *
from utils.svm import *
from utils.kmeans import *
from utils.preprocess import *

# data files (constants)
ftest = './dat/testTriplets.txt' # test data
ftrain = './dat/txTripletsCounts.txt' # train data


# run main program
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description = 'Predicting Bitcoin transactions')
    parser.add_argument('-t', '--threshold', type=int, help = '1 to use binary transacted vs not transacted, 0 to use # of transactions',  choices = [0,1], required=True)
    parser.add_argument('-S', '--SVD', action="store_true", help= 'run SVD')
    parser.add_argument('-sn', '--num_sing', type=int, help= 'number of singular values for SVD')
    parser.add_argument('-SVM', '--SVM', action="store_true", help= 'run SVM')
    parser.add_argument('-nzero', '--num_zeroes', type=int, help= 'number of zeroes to predict for SVM')
    """
    parser.add_argument('-K1', '--kmeans1', action="store_true", help= 'run kmeans algorithm 1')
    parser.add_argument('-K2', '--kmeans2', action="store_true", help= 'run kmeans algorithm 2')
    parser.add_argument('-K3', '--kmeans3', action="store_true", help= 'run kmeans algorithm 3')
    parser.add_argument('-K4', '--kmeans4', action="store_true", help= 'run kmeans algorithm 4')
    parser.add_argument('-nc', '--num_clusters', type=int, help= 'number of clusters for kmeans')
    """


    args = parser.parse_args()
    print args
    
    # load data
    train_data = ReadData(ftrain)
    test_data = ReadData(ftest)

    # convert training data into sparse matrix format
    train_data = PreprocessData(train_data, threshold = args.threshold)

    # perform predictions
    if args.SVD:
        RunSVD(train_data, test_data, num_sing = args.num_sing, threshold=args.threshold)
    if args.SVM:
        RunSVM(train_data, test_data, num_zeroes=args.num_zeroes)
    """
    if args.kmeans1:
        RunKmeans(train_data, test_data, algorithmID = 1, num_clusters = args.num_clusters, threshold = args.threshold)
    if args.kmeans2:
        RunKmeans(train_data, test_data, algorithmID = 2, num_clusters = args.num_clusters, threshold = args.threshold)
    if args.kmeans3:
        RunKmeans(train_data, test_data, algorithmID = 3, num_clusters = args.num_clusters, threshold = args.threshold)
    if args.kmeans4:
        RunKmeans(train_data, test_data, algorithmID = 4, num_clusters = args.num_clusters, threshold = args.threshold)
    """
