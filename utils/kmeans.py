#!/usr/bin/python
# 
# Run various clustering algorithms based on k-means
#
# Author: Chaney Lin
# Date: March 2015
#

import time
import numpy as np
from preprocess import *

def write_kmeans(labels, centroids, num_clusters):
    flabels = 'kmeans_' + str(num_clusters) + '_labels.txt'
    with open(flabels, 'w') as f:
        for l in labels:
            f.write(str(l) + '\n')
            
    fcentroids = 'kmeans_' + str(num_clusters) + '_centroids.txt'
    with open(flabels, 'w') as f:
        for c in centroids:
            f.write(str(c) + '\n')

def cluster_method1(test_data, labels, centers, cluster_num, threshold):
    """
    cluster method 1 (predicts and writes data)
    assigns probability of (i,j) as position j of center of i
    """
    len_test = len(test_data)
    start = time.time()
    kmeans_pred = []
    for i in range(len_test):
        row = test_data[i][0]
        col = test_data[i][1]
        i_label = labels[row]
        kmeans_pred.append(centers[i_label][col])
    end = time.time()
    print 'predicting time (method 1): ', end-start, 'sec'
    
    if threshold == 1:
        fname = 'th_kmeans_cluster1_' + str(cluster_num) + '.txt'
    else:
        fname = 'kmeans_cluster1_' + str(cluster_num) + '.txt'

    WritePredictions(kmeans_pred,fname)
    
    return kmeans_pred

def cluster_method2(train_data, test_data, labels, centers, cluster_num, threshold):
    """
    cluster method 2 (predicts and writes data)
    for each i, looks at clusters of labels of trade partners (trade_cluster)
    if j is in trade_cluster, then they trade
    """

    start = time.time()
    kmeans_pred = []
    
    label_cluster = {}

    len_test = len(test_data)

    for i in range(len_test):
        row = test_data[i][0]
        col = test_data[i][1]
        
        if row not in label_cluster:
            label_cluster[row] = np.sort(np.array(list(set([labels[x] for x in train_data[row,].nonzero()[1]]))))
        
        if labels[col] in label_cluster[row]:
            kmeans_pred.append(1)
        else:
            kmeans_pred.append(0)
    end = time.time()
    print 'predicting time (method 2): ', end-start, 'sec'

    if threshold == 1:
        fname = 'th_kmeans_cluster2_' + str(cluster_num) + '.txt'
    else:
        fname = 'kmeans_cluster2_' + str(cluster_num) + '.txt'

    WritePredictions(kmeans_pred,fname)
    
    return kmeans_pred

def cluster_method3(train_data, test_data, labels, centers, cluster_num, threshold):
    """
    cluster method 3 (predicts and writes data)
    for each i in cluster C_i, C_i will trade with all of
    clusters of i's trade partners C_j
    """
    start = time.time()
    kmeans_pred = []
    
    n = len(labels)
    len_test = len(test_data)
    
    label_cluster = [[] for x in range(cluster_num)]
    
    for x in range(n):
        label_i = labels[x]
        trade_cluster = [labels[y] for y in train_data[x,].nonzero()[1]]
        label_cluster[label_i] += trade_cluster

    label_cluster = np.array([np.sort(np.array(list(set(row)))) for row in label_cluster])

    for i in range(len_test):
        row = test_data[i][0]
        col = test_data[i][1]
        
        if labels[col] in label_cluster[labels[row]]:
            kmeans_pred.append(1)
        else:
            kmeans_pred.append(0)
    end = time.time()
    print 'predicting time (method 3): ', end-start, 'sec'

    if threshold == 1:
        fname = 'th_kmeans_cluster3_' + str(cluster_num) + '.txt'
    else:
        fname = 'kmeans_cluster3_' + str(cluster_num) + '.txt'

    WritePredictions(kmeans_pred,fname)
    
    return kmeans_pred


def cluster_method4(train_data, test_data, labels, centers, cluster_num, threshold):
    """
    cluster method 4 (predicts and writes data)
    for each i in cluster C_i, C_i will trade with all of
    i's trade partners
    """
    start = time.time()
    kmeans_pred = []
    
    n = len(labels)
    
    label_cluster = [[] for x in range(cluster_num)]
    
    for x in range(n):
        label_i = labels[x]
        partners = [y for y in train_data[x,].nonzero()[1]]
        label_cluster[label_i] += partners

    label_cluster = np.array([np.sort(np.array(list(set(row)))) for row in label_cluster])

    for i in range(len_test):
        if i%1000 == 0: print i
        row = test_data[i][0]
        col = test_data[i][1]
        
        if col in label_cluster[labels[row]]:
            kmeans_pred.append(1)
        else:
            kmeans_pred.append(0)
    end = time.time()
    print 'predicting time (method 4): ', end-start, 'sec'

    if threshold == 1:
        fname = 'th_kmeans_cluster4_' + str(cluster_num) + '.txt'
    else:
        fname = 'kmeans_cluster4_' + str(cluster_num) + '.txt'

    WritePredictions(kmeans_pred,fname)
    
    return kmeans_pred


def RunKmeans(train_data, test_data, algorithmID, num_clusters = 100, threshold = 1):
    """
    run K-means

    Inputs:
    train_data      -- training data
    test_data       -- testing data
    algorithmID     -- which of the four clustering algorithms to use
    num_clusters    -- number of clusters (default 100)
    threshold       --  1: use binary (pair has transacted vs not transacted)
                        0: use raw (uses the actual number of transactions)

    Returns
    array of predictions
    """
    from sklearn import cluster

    # fitting
    start = time.time()

    kmean = cluster.MiniBatchKMeans(n_clusters = num_clusters, batch_size=1000, n_init = 100)
    labels = kmean.fit_predict(train_data)
    centers = kmean.cluster_centers_

    end = time.time()

    print 'fitting time: ', end-start, 'sec'

    # predicting using clustering methods

    if algorithmID == 1:
        kmeans_pred = cluster_method1(test_data, labels, centers, num_clusters, threshold)

    if algorithmID == 2:
        kmeans_pred = cluster_method2(train_data, test_data, labels, centers, num_clusters, threshold)
        
    if algorithmID == 3:
        kmeans_pred = cluster_method3(train_data, test_data, labels, centers, num_clusters, threshold)
        
    if algorithmID == 4:
        kmeans_pred = cluster_method4(train_data, test_data, labels, centers, num_clusters, threshold)