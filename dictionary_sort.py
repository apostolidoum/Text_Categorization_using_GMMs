# -*- coding: utf-8 -*-
"""
Created on Mon May 14 18:35:01 2018

@author: maria
"""

import numpy as np
"""
Sort inputs according to importance of the terms.
Returns : TF_sorted, IDF_sorted, Dictionary_sorted

sum over TF columns to find a value of the term over all documents-classes
find TF-IDF value for each term - that is considered to be the importance of 
the term
sort dictionary terms according to their importance
"""

def dictionary_sort(TF_table, IDF_table, Dictionary):
    #--- use this code to sort according to TFIDF values ---#
    TF_zipped = np.mean(TF_table,axis = 0)
    importance = TF_zipped * IDF_table     
    
    # sort all matrices according to the importance list
    sorted_indexes = np.argsort(importance)
    sorted_indexes = sorted_indexes[::-1] #reverse indexes -> descending order
    #-------------------------------------------------------#

    #--- use this code to sort according to IDF values only ---#
    #--- Perform terribly because rare words (on their own are useless # 
#    sorted_indexes = np.argsort(IDF_table)
#    sorted_indexes = sorted_indexes[::-1] #reverse indexes -> descending order
    #----------------------------------------------------------#
    
    # permute collumns according to sorted indexes
    TF_table = TF_table[:,sorted_indexes]

    IDF_table = IDF_table[sorted_indexes]
    
    Dictionary_new = [Dictionary[i] for i in sorted_indexes]
    
 
    return [TF_table, IDF_table, Dictionary_new]
    
    
"""
Testing
uncomment this code to test with a simple example
"""
## Let's try a trivial example
## Suppose the following inputs
## TF = [[1,4,7,3],
##       [3,6,1,3],
##       [4,6,1,2]]
##
## IDF = [1,3,5,1]
##
## Dictionary = ['word0','word1','word2','word3']
##
## The reduced TF will be TF_zipped = [2.7,5.3,3,2.7]
## The importancy (TF_zipped * IDF) will be [2.7,15.9,15,2.7]
## From most to least important (descending order)  we get  15.9 15 2.7 2.7 
## that is a permutation of the form 1 2 3 0 
##
## and the sorted outputs according to the importance 
## Dictionary [word1, word2, word3, word0]        
#
#TF = np.array([[1,4,7,3],[3,6,1,3],[4,6,1,2]])
#IDF = np.array([1,3,5,1])
#Dictionary = ['word0','word1','word2','word3']
#
#[TF_sorted, IDF_sorted, Dict_sorted] = dictionary_sort(TF,IDF,Dictionary)
#print("TF : \n" ,TF_sorted)
#print("IDF : " ,IDF_sorted)
#print("Dict : " , Dict_sorted)
