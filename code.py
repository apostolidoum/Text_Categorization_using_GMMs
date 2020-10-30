#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 18:08:50 2018

Technical University of Crete

Course   :  Pattern Recognition
            Spring 2017-2018
Project  :  15.2
            Call Routing System                    

@authors :  Apostolidou Maria (2012030114)
            Kyparissas Nikolas (2012030112)
"""
import sys
import numpy as np
from dictionary_sort import *
from tfidf import *
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
import csv

'''
Inputs from console/bash
The parameters
'''
# downsize the dictionary before svd
# the dictionary (and tf idf tables) are sorted according to importance of the words
# the #discard least important words are deleted to reduce the dimensionality 
# of the data at a first level 
discard = int(sys.argv[1])
# number of components for the svd reduction
# #svd_components must be < #features
svd_components = int(sys.argv[2])
# number of gaussian components for the gmms
# gmm_components must be <= #samples
gmm_components = int(sys.argv[3])

"""
Create the dictionary
"""
dictionary = []
labels = []
with open('data/final.train') as train_file :
    """
    Create the dictionary
    """
    num_of_lines = 0 # count the rows / document number
    training_data = train_file.readlines()
    for line in training_data:
        num_of_lines += 1
        words = line.split()
        words_no_class = words[1:] # discard the label of the sample 
        labels.append(words[0])
        for word in words_no_class:
            if word not in dictionary:
                dictionary.append(word)
train_file.closed    

'''
Find the term document matrix from the training data
'''  
term_document = term_doc('data/final.train',num_of_lines,dictionary)

"""
Find the term frequencies from the training data
"""
TF = tf(num_of_lines,dictionary,term_document)


"""
Inverse document Frequency IDF from the training data
"""
IDF = idf(num_of_lines,dictionary,term_document)


"""
Sort the dictionary
"""
# Sort according to TF-IDF values
# Sort over TF value for all our classes (documents)
# Maybe this is not the best approach to the problem

[TF,IDF,dictionary] = dictionary_sort(TF,IDF,dictionary)

"""
Downsize features
Discart the least important 
"""
# discard that many elements from the end of TF IDF and dictionary
if discard > 0 :
    TF = TF[:,:-discard]
    IDF = IDF[:-discard]
    dictionary = dictionary[:-discard]

'''
Create the TF-IDF MATRIX from the training data
to be used with svd and gmms
'''
TFIDF = tfidf(TF,IDF)

'''
Singular Value Decomposition
'''
svd = TruncatedSVD(n_components=svd_components)
TFIDFsvd = svd.fit_transform(TFIDF)

'''
extract data for each class separately
GMMs will be trained separately on each classes TFIDF samples
'''
TFIDF_class = []
for class_num in range(1,16):
    TFIDF_class.append(samples_from_class(TFIDFsvd,class_num,labels))


'''
GMM training
We train #classes = 15 GMMS to estimate the distribution of the features 
Each row of the TFIDFsummed is a feature vector on which we train a GMM
'''
GMMS = []
for class_num in range(1,16):
    # ATTENTION: indexes of TF go from 0 - 14
    #            whereas the class numbers go from 1 - 15
    GMMS.append(GaussianMixture(n_components= gmm_components).fit(TFIDF_class[class_num-1]))


'''
Testing
'''
test_labels = []
with open('data/final.test') as test_file :
    testsamples = test_file.readlines()
    num_of_test_data = 0; #count rows
    for line in testsamples:
        num_of_test_data += 1
        testwords = line.split()
        test_labels.append(testwords[0])
test_file.closed   


'''
Find the term document matrix from the test data
'''  
test_term_document = term_doc('data/final.test',num_of_test_data,dictionary)

"""
Find the term frequencies from the test data
"""
test_TF = tf(num_of_test_data,dictionary,test_term_document)


"""
Inverse document Frequency IDF from the test data
"""
test_IDF = idf(num_of_test_data,dictionary,test_term_document)

'''
Create the TF-IDF MATRIX from the test data
'''
test_TFIDF = tfidf(test_TF,test_IDF)

'''
Dimensionality Reduction for the test samples
'''
test_TFIDFsvd = svd.transform(test_TFIDF)

'''
For each test sample find the loglikelihood probabilities for
each classe's GMM
'''
# contains the loglikelihoods for each GMM => class
classloglike = []
# ATTENTION: indexes of TF go from 0 - 14
#            whereas the class numbers go from 1 - 15
for class_idx in range(15):
    classloglike.append(GMMS[class_idx].score_samples(test_TFIDFsvd))

'''
ML classify
Classify each test sample to the class for which
the loglikelihood probability (that the sample came from the GMM that 
represents this class) is maximum
'''
# contains the decision indexes
classified_idx = np.array(len(test_labels))
classified_idx = np.argmax(classloglike,axis=0)
    
# convert the decision indexes to labels
# ATTENTION: indexes of TF go from 0 - 14
#            whereas the class numbers go from 1 - 15
classified_labels = [x+1 for x in classified_idx]

'''
Accuracy
compare the actual test labels with the labels that we decided during 
classification and find the accuracy of the classification task
''' 
# used to save the accuracy level for each class
class_accuracy = np.zeros(15)
total_class_instances = np.zeros(15)
accurate = 0  
# ATTENTION : the test_labels are a string
#             while the classified_labels are a list of int
#             convert int to str to compare
for classified,original in zip(map(str,classified_labels),test_labels):
    total_class_instances[int(original)-1] += 1
    if classified == original : 
        accurate += 1
        class_accuracy[int(original)-1] += 1

percentage = accurate/len(classified_labels)
print("Number of accurate classificiations: ",end="")
print(accurate)
print("Accuracy level: ",end="")
percentage = percentage*100
print(percentage,end="")
print(" %")
print("Accuracy results for each class: ")
for i in range(15):
    print("class ",end="" )
    print(i,end="")
    print(" : ", end="")
    print(class_accuracy[i]/total_class_instances[i]*100,end="")
    print(" %")


# write accuracy results to csv file 
# for accuracy analysis
with open('results.csv','a',newline='') as csvfile:
    writer = csv.writer(csvfile,delimiter=';')
    writer.writerow([discard,svd_components,gmm_components,percentage])



