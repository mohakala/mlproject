# -*- coding: utf-8 -*-
"""
Data exploratin and machine learning
Created on Sun Apr  2 09:41:42 2017
@author: mikko hakala
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class mlproject(object):
    """
    Items needed to carry out basic data exploration and machine learning
    Uses pandas dataframes
    """
    
    def getData(self, path):
        df = pd.read_csv(path, delimiter=",")
        self.df = df


    def plot(self, x, y):
        plt.figure()
        plt.scatter(x, y)        
        #plt.axis('tight')
        #plt.ylabel('R2')
        #plt.xlabel('neighbors')
        #plt.legend()
        plt.show()


    def addHeader(self, header):
        """
        Add header to the project
        """
        self.header=header

        
    def examine(self):
        """
        Show basic information of the dataframe
        """
        print(self.header)
        print(self.df.head())
        print(self.df.describe())

                
    def modifyData(self, func):
        """ 
        Modify df using function func
        """
        self.df = func(self.df)
        
        
    def randomizeRows(self, seed=None):
        """
        Randomize the rows of the dataframe
        """
        np.random.seed(seed)
        self.df=self.df.reindex(np.random.permutation(self.df.index))
        self.df=self.df.reset_index(drop=True)


    def head(self):
        """
        Show a few lines of the dataframe
        """
        print(self.df.head())


    def rename(self, columns):
        """
        Rename columns
        """
        self.df = self.df.rename(columns = columns)

    
    def score(self, model):
        model.fit(self.Xtrain, self.ytrain)
        score_train = model.score(self.Xtrain, self.ytrain)
        score_val = model.score(self.Xval, self.yval)
        #score_cross_val = cross_val_score(knn, Xtrain, ytrain, cv=5)
        print('\nScore (R2 on the (reduced) training set):\t', score_train)
        print('Validation score (R2):\t\t', score_val)
        #print('Cross-validation score (R2), mean:\t', score_cross_val.mean(), '+-', score_cross_val.std()*2, ('(95% confid. interval)'))


    def set(self, Xtrain, ytrain, Xval, yval, Xtest, ytest):
        """
        Set X and y
        """
        self.Xtrain = Xtrain
        self.Xval = Xval
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.yval = yval
        self.ytest = ytest
        
        

