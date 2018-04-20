# Predicting Bitcoin transactions

We analyze the blockchain and try to predict whether two individuals who have never exchanged Bitcoin will later transact

This was an assignment for Princeton's COS424 course (Fundamentals of Machine Learning) in March 2015

For further details, please refer to the [writeup](./writeup/report_bitcoin%20predictions.pdf)

# Data description

The data is derived from the Bitcoin blockchain

The **training set** comprises 3.3M transactions among ~440K unique addresses, for the year March 2012-13

The **test set** consists of 10K addresses which did not transact with anyone during March 2012-13; 1K of them later transacted with at least one of the individuals in the training set

# Models

We performed predictions using the following models

1. singular value decomposition (SVD)

2. k-means clustering (four versions)

3. support vector machine (SVM)

# Work-in-progress

A partial collection of code has been updated; still need to clean up the rest
