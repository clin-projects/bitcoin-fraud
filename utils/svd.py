#!/usr/bin/python
# 
# Run singular value decomposition
# for documentation, see:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html
#
# Author: Chaney Lin
# Date: March 2015
#

from preprocess import *
import time

def RunSVD(train_data, test_data, num_sing = 10, threshold = 0):
    """
    run SVD

    Inputs:
    train_data      -- training data
    test_data       -- testing data
    num_sing        -- number of singular values to keep
    threshold       --  1: use binary (pair has transacted vs not transacted)
                        0: use raw (uses the actual number of transactions)

    Returns
    array of predictions
    """

    from scipy.sparse.linalg import svds

    print 'SVD with', num_sing, 'singular values'
    start = time.time()
    u,s,vt = svds(train_data, num_sing)
    end = time.time()

    print 'fitting time: ', end-start, 'sec'

    start = time.time()
    svd_pred = []
    us = np.dot(u,np.diag(s))
    vtt = np.transpose(vt)
    for i in range(len(test_data)):
        row = test_data[i][0]
        col = test_data[i][1]
        svd_pred.append(np.dot(us[row],vtt[col]))
        
    end = time.time()
    
    print 'predicting time: ', end-start, 'sec'

    if threshold == 1:
        fname = 'th_svd' + str(num_sing) + '.txt'
    else:
        fname = 'svd' + str(num_sing) + '.txt'
    WritePredictions(svd_pred,fname)
    return svd_pred