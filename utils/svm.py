#!/usr/bin/python
# 
# Run support vector machine
#
# Author: Chaney Lin
# Date: April 2018
#

import numpy as np
import time
import random
from sklearn import svm
import scipy

start = time.time()

def RunSVM(train_data, test_data, num_zeroes=20):
    """
    run SVM

    Inputs:
    train_data      -- training data
    test_data       -- testing data
    
    Returns
    array of predictions
    """

    
    clf = svm.SVC(kernel = 'linear')

    a = train_data.toarray()

    def predict(i, index, num_zeroes=20):
        """
        i       -- index of sender
        index   -- index of receiver
        i is in the "list_sender"(list) and for each i, the receiver is in the list "index_list"
        num_zeroes is the number of randomly chosen zero values to predict
        """
    
        trade_index_array = np.where(a[:,index]>0)

        # Need eliminate a[k,index]
        trade_array = []

        for k in trade_index_array[0]:
            b = np.delete(a[k,:], index)
            trade_array.append(b)
            non_trade_array = []

            for temp in xrange(10):
                k = random.randint(0, 444065)
                if k not in trade_index_array[0] and temp < number:

                    non_trade_array.append(np.delete(a[k,:],index))
                    temp = temp+1

        ylabel = np.concatenate((np.ones(len(trade_array)),np.zeros(len(non_trade_array))),axis = 0)
        xlabel = np.concatenate((trade_array,non_trade_array),axis = 0)
        clf.fit(xlabel,ylabel)

        #eliminate a[i, index]
        return clf.predict(np.delete(a[i,:],index))

    svm_pred = []
    len_test = len(test_data)
    for i in range(len_test):
        row = test_data[i][0]
        col = test_data[i][1]
        svm_pred.append(predict(row,col,num_zeroes=num_zeroes))
    return svm_pred