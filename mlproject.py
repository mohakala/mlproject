# -*- coding: utf-8 -*-
"""
Data exploratin and machine learning
Created on Sun Apr  2 09:41:42 2017
@author: mikko hakala
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score



class mlproject(object):
    """
    Items needed to carry out basic data exploration and machine learning
    Uses pandas dataframes
    """
    
    def debug(self, value=True):
        self.debug=value

    
    def getData(self, path, delimiter=',', header='infer'):
        df = pd.read_csv(path, delimiter=delimiter, header=header)
        self.df = df
        

    def missingValues(self):       
        """
        Report missing values
        """
        self.missingvalues=self.df.apply(lambda x: sum(x.isnull()),axis=0)
        return self.missingvalues


    def fillMissingCategorical(self, label):
        """
        Fill missing values of column 'label' with average value
        """
        self.df[label].fillna(self.df[label].value_counts().index[0], inplace=True)


    def plot(self, x, y):
        plt.figure()
        plt.scatter(x, y)        
        #plt.axis('tight')
        #plt.ylabel('R2')
        #plt.xlabel('neighbors')
        #plt.legend()
        plt.show()


    def plot_new(self, i, j, xlabel=" ", ylabel=" "):
        plt.figure()
        plt.plot(self.df[i], self.df[j], 'o')        
        #plt.axis('tight')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
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
        try:
            print(self.header)
        except AttributeError:
            pass
        
        print(self.df.head())
        print(self.df.describe())



    def dropna(self):
        self.df = self.df.dropna()
        
                
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
        """
        Trains the model
        Calculates training, validation and cross-validation scores
        """
        self.model = model
        model.fit(self.Xtrain, self.ytrain)
        self.score_train = model.score(self.Xtrain, self.ytrain)

        # If size of validation set larger than 1, assume validation to be done 
        if len(self.yval) > 1:
            self.score_val = model.score(self.Xval, self.yval)
            self.score_cross_val = cross_val_score(model, self.Xtrain, self.ytrain, cv=5)
        else:
            pass

    
    def score_print(self):
        np.set_printoptions(precision=3)
        trainscore = np.round(self.score_train, 2)
        print('\nScore (R2 on the (reduced) training set):\t', trainscore)

        # If size of validation set larger than 1, assume validation to be done 
        if len(self.yval) > 1:
            valscore  = np.round(self.score_val, 2)
            cvalscore = np.round(self.score_cross_val.mean(), 2)
            cvalscore_std = np.round(self.score_cross_val.std()*2, 2)
            print('-Validation score (R2):\t\t\t', valscore )
            print('-Cross-validation score (R2), mean:\t\t', cvalscore , '+-', cvalscore_std , ('(95% confid. interval)'))


    def print_coef(self):
        np.set_printoptions(precision=3)
        try: 
            print('.. intercept:', self.model.intercept_)
            print('.. params: ', self.model.coef_)
            return(self.model.intercept_, self.model.coef_)
        except:
            pass


    def set_old(self, Xtrain, ytrain, Xval, yval, Xtest, ytest):
        """
        Set X and y
        """
        self.Xtrain = Xtrain
        self.Xval = Xval
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.yval = yval
        self.ytest = ytest

        
    def set_new(self, target, features, ind):
        """
        Set X and y, makes the train, validate, test split
 
        Makes roughly this:
       
        y = ml.df[[yname]].values 
        X = ml.df[features].values
       
        Xtrain = X[:234, :]
        ytrain = y[:234].reshape(-1,1)
        Xval   = X[235:294, :]
        yval   = y[235:294].reshape(-1,1)
        Xtest  = X[295:, :]
        ytest  = y[295:].reshape(-1,1)

        """
        self.X = self.df[features].values
        self.y = self.df[target].values
        assert ind[0] <= len(self.y), "Error: ind[0] for max index for training data must be equal or smaller than len(target)" 

        if(ind[0]==len(self.y)):
            print('Note: ind[0]=', ind[0], ', using all the data for training (no validation data)')


        self.Xtrain = self.X[:ind[0], :]
        self.ytrain = self.y[:ind[0]].reshape(-1,1)

        self.Xval   = self.X[ind[0]+1:ind[1], :]
        self.yval   = self.y[ind[0]+1:ind[1]].reshape(-1,1)

        self.Xtest  = self.X[ind[1]+1:, :]
        self.ytest  = self.y[ind[1]+1:].reshape(-1,1)
        
        if(self.debug):
            print('target:', self.y)
            print('ind, len y:', ind, len(self.y))
            print('ytrain:', self.ytrain)

                    
        
    def predict(self, X):
        X = np.array(X).reshape(1,-1)
        return self.model.predict(X)
        
        

