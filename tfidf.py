# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:02:19 2018

@author: maria
"""
"""
TF IDF FUNCTIONS
"""
import numpy as np

"""
Creating the term-document matrix.
Training data at data/final.train
Testing data at data/final.test
Columns of the matrix are all the differend words that appear 
        in our training data.
Rows correspond to the training samples / documents
Each element (row i ,column j) in this matrix is the number of times we have
        encoundered in sample sentence i the word j
Inputs : filename -> file with the data 
         sizeofdata -> #rows in file, aka #documents
         dictionary we have created from train dataset
"""
def term_doc(filename,size_of_data,dictionary):
    with open(filename) as data_file :
        data = data_file.readlines()
        # initialize term_document matrix with zeros
        term_document = np.zeros((size_of_data,len(dictionary)),dtype=int)
        
        i = 0 # count parsed lines
        for line in data:
            words = line.split()
            words_no_class = words[1:] # discard the label of the sample 
           
            for word in words_no_class:
                if word in dictionary:
                    index = dictionary.index(word)
                    term_document[i][index] += 1
                
            i += 1
    data_file.closed
    return term_document
 
"""
Find the term frequencies
Creates a tf matrix from a term document matrix
"""   
def tf(size_of_data,dictionary,term_document):
    TF = np.zeros((size_of_data,len(dictionary)))
   
    
    # normalize TF values
    # divide by sum of row
    # Handle Exception : if denominator = sum_of_row = 0 then
    #                    keep value of the nominator = TF[i,:]
   
    sum_of_row = np.sum(term_document,axis=1)
    for i in range(size_of_data):
        TF[i,:] = np.divide(term_document[i,:],sum_of_row[i],out=np.zeros_like(TF[i,:]), where = sum_of_row[i] != 0)
    return TF

"""
Inverse document Frequency IDF
"""
def idf(size_of_data,dictionary,term_document):
        IDF = np.zeros(len(dictionary))
        number_of_documents_with_term = np.zeros((len(dictionary)),dtype=int)        
        # COUNT (not sum ) how many documents have this term
        # count non zero values of term_documents per column => term
        # sum() counts
        for j  in range(len(dictionary)):
            number_of_documents_with_term[j] = (term_document[:,j]!=0).sum()
        
        # IDF matrix (1 x size_of_dictionary)
        #  ln(total number of documents/total number of documents with term t in it) 
        # because we have the ln 
        # for the case of zero division output a value of 1 so that
        # ln1 = 0 in the final IDF
        IDF = np.log(np.divide(size_of_data,number_of_documents_with_term,out=np.ones_like(IDF), where = number_of_documents_with_term != 0))
        return IDF
    

"""
TF IDF 
"""
def tfidf(TF,IDF):
    TFIDF = np.zeros(TF.shape)
    # len(IDF) =  len(dictionary)
    for i in range(len(IDF)):
        TFIDF[:,i] = TF[:,i]*IDF[i]
    return TFIDF
    
    
'''
Collect data of class
From a TFIDF table find all the samples that belong to a specific class
input : class_num -> a number fron 1 to 15 that corresponds to the class
        TFIDF -> the table
        labels -> the class information of the TFIDF 
returns a table with these samples
'''  
def samples_from_class(TFIDF,class_num,labels):
    # append the labels on the TFIDF table in the first column, 
    # this makes the extraction of specific rows easier
    classTFIDF = np.insert(TFIDF,0,labels,axis=1)
    # collect rows that have as their first element the value of the class_num
    TFIDF_class_samples = classTFIDF[np.where(classTFIDF[:,0]==class_num)]
    # delete the first column that contained the class label information
    TFIDF_class_final = np.delete(TFIDF_class_samples,0,axis=1)
    return TFIDF_class_final
    
    
    
    
    
    
    
