#!/usr/bin/python
# 
# Preprocessing functions
#
# Author: Chaney Lin
# Date: March 2015
#

import csv
import numpy as np

def ReadData(fin):
    """
    read in transaction data where rows contain three integers i, j, c_ij
    where i is sender ID, j is receiver ID, and c_ij is the number of times the pair
    (i,j) has transacted, for only those where c_ij is nonzero

    return numpy array containing data (integers)
    """

    print 'reading in data from ' + fin
    dat = []
    with open(fin, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            dat.append([int(x) for x in row[0].split()])
    return np.array(dat)

def ConvertToSparse(A):
    """
    converts matrix into sparse matrix format
    """
    from scipy.sparse import csr_matrix

    return csr_matrix((A[:,2].astype(float),(A[:,0],A[:,1])))

def PreprocessData(A, threshold = 0):
    """
    Preprocesses the training data; converts to sparse matrix format, and performs
    thresholding (if threshold == 1)

    Inputs:
    A               -- input matrix where rows are of form i, j, c_ij
    threshold       --  1: use binary (pair has transacted vs not transacted)
                        0: use raw (uses the actual number of transactions)

    Returns processed A
    """
    if threshold:
        thresholded = A
        thresholded[:,2] = (thresholded[:,2] > 0)
        return ConvertToSparse(thresholded)
    else:
        return ConvertToSparse(A)

def WritePredictions(dat_pred, fout):
    """
    output predictions (dat_pred) to file fout
    """
    with open(fout,'w') as f:
        for x in dat_pred:
            f.write(str(x) + '\n')