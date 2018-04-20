# Predicting Bitcoin transactions

Assignment submitted for Princeton's COS424 course (Fundamentals of Machine Learning) in March 2015

Refer to [writeup](./writeup/report_bitcoin%20predictions.pdf) for further details

The goal is to be able to predict whether two individuals will exchange Bitcoin, based on an analysis of the blockchain

# Data description

The data is derived from the Bitcoin blockchain.

The **training set** comprises 3.3M transactions among ~440K unique addresses, for the year March 2012-13. Refer to the addresses in the training set as the _training addresses_.

The **test set** consists of 10K addresses which did not transact with the training addresses during March 2012-13; 1K of them later transacted with at least one of the training addresses between March 2013-14

# Models

We performed predictions using the following models

1. singular value decomposition (SVD)

2. k-means clustering (four versions)

3. support vector machine (SVM)

# Work-in-progress

A partial collection of code has been updated; still need to clean up the rest
