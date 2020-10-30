#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun June 3 15:11:27 2018

@author: maria

The program imports our data into Panda's dataframes. 
Then separates the cases and plots them. 
"""
import pandas as pd  # for data analysis
import numpy as np  # for numerical things
import matplotlib.pyplot as plt  # for plotting

# Load the data from the file into a Pandas dataframe
resultsFile = 'results/results.csv' 
# import the data from csv into panda's dataframe
resultsDF = pd.read_csv(resultsFile, sep=';', header=None)  
# name the different columns of the dataframe
resultsDF.columns = ['N deleted', 'SVD', 'GMM comp', 'Accuracy'] 
# Plot for the different dictionary reduction numbers
for reductionNumber in np.arange(500, 3500, 500):
    plotData = resultsDF[resultsDF['N deleted'] == reductionNumber]  # selects the data corresponding to the particular dic reduction number
    ax = plt.gca()
    legendNames = []  # sets an empty array for creation of the plot legend
    for SVD in np.arange(15, 55, 5):
        legendNames.append('SVD '+str(SVD))  # creates the plot legend string list
    for SVD in np.arange(15, 55, 5):
        plotData.loc[(plotData['SVD'] == SVD)].plot(x='GMM comp', y='Accuracy', ax=ax)  # plots for the specific case of dic reduction number
    # Set a few parameters for the plot
    ax.set_title('Dictionary reduction size '+ str(reductionNumber))
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Number of Gaussian Mixture Model components')
    ax.legend(legendNames)
    # Save the plot, close the figure and move to the next case
    plt.savefig("figs/reduce%d.png"%reductionNumber, dpi=300)
    plt.close()
