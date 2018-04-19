# Predicting Bitcoin transactions

Assignment submitted for Princeton's COS424 course (Fundamentals of Machine Learning) in March 2015

Refer to [writeup](./writeup/report_bitcoin%20predictions.pdf) for further details

# Data description

Training set contains 3.3M transactions among ~440K unique addresses, for the year between March 2012 to March 2013

Test set contains 10K addresses which did not interact with the individuals in the training set during March 2012-13; 1K of them later interacted with the training set individuals between March 2013-14

The objective is to predict whether the pairs in the test data traded or not with the individuals in the training set

# Models

We used the following models

1. singular value decomposition (SVD)

2. k-means clustering (four versions)

3. support vector machine (SVM)

# Work-in-progress

Have uploaded a partial collection of code; still need to clean up the rest
