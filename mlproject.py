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
    
    def __init__(self, df=None):
        """
        Use input pandas dataframe if exists
        """
        if df is not None:
            self.df = df
    
    def debug(self, value=False):
        self.debug=value
        if self.debug:
            print('**debug set to True**')

    def getData(self, path, delimiter=',', header='infer'):
        df = pd.read_csv(path, delimiter=delimiter, header=header)
        self.df = df

    def dropColumn(self, column):
        """
        Drops a column from the pandas dataframe
        """
        self.df = self.df.drop(column, 1)
        
    def normalize(self):
        self.means = self.df.mean()
        self.ranges = (self.df.max() - self.df.min()) 
        self.df =  (self.df - self.df.mean()) / (self.df.max() - self.df.min())

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

    def plot(self, x, y=None):
        plt.figure()
        if y is None:
            plt.plot(x)
        else:
            plt.scatter(x, y)        
        #plt.axis('tight')
        #plt.ylabel('R2')
        #plt.xlabel('neighbors')
        #plt.legend()
        plt.show()

    def plot_new(self, i, j=None, xlabel=" ", ylabel=" "):
        plt.figure()
        if j is None:
            plt.plot(self.df[i], 'o')
        else:
            plt.plot(self.df[i], self.df[j], 'o')        
        #plt.axis('tight')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.xlabel('neighbors')
        #plt.legend()
        plt.show()

    def plot_date(self, i, j, xlabel=" ", ylabel=" "):
        plt.figure()
        plt.plot_date(self.df[i], self.df[j])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def addHeader(self, header):
        self.header=header
        
    def examine(self):
        """
        Show basic information of the dataframe
        """
        try:
            print(self.header)
        except AttributeError:
            pass        
        print('Shape of df:', self.df.shape)
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

    def head(self, n=6):
        print(self.df.head(n))

    def rename(self, columns):
        """
        Rename columns
        """
        self.df = self.df.rename(columns = columns)
            
    def score(self, model, cv=5, iprint=1, printTestScore=False):
        """
        Trains the model
        Calculates training, validation, cross-validation and test scores
        cv: cross-validation folds
        """
        self.model = model
        self.cv = cv
        
        model.fit(self.Xtrain, self.ytrain)
        self.score_train = model.score(self.Xtrain, self.ytrain)
        self.score_cross_val = cross_val_score(model, self.Xtrain, self.ytrain, cv=cv)

        if self.isVal:
            self.score_val = model.score(self.Xval, self.yval)
        if self.isTest:
            self.score_test = model.score(self.Xtest, self.ytest)

        self.score_print(iprint, printTestScore)
        return self.score_cross_val.mean()

    
    def score_print(self, iprint=1, printTestScore=False):
        np.set_printoptions(precision=3)
        if iprint > 2:
            trainscore = np.round(self.score_train, 3)
            print('\nTraining score (R2 on training set):\t', trainscore)

        if iprint > 0:
            cvalscore = np.round(self.score_cross_val.mean(), 3)
            cvalscore_std = np.round(self.score_cross_val.std()*2, 3)
            print('Cross-validation score (R2), folds=', self.cv, 'mean:\t', cvalscore , '+-', cvalscore_std , ('(standard dev.)'))

        if self.isVal and iprint > 1:
            valscore  = np.round(self.score_val, 3)
            print('Validation score (R2):\t\t\t', valscore )

        if (self.isTest and printTestScore) or (self.isTest and iprint > 3):
            print('Test score:\t', self.score_test)
            



    def print_coef(self):
        np.set_printoptions(precision=3)
        try: 
            print('.. intercept:', self.model.intercept_)
            print('.. params: ', self.model.coef_)
            return(self.model.intercept_, self.model.coef_)
        except:
            pass

    def set_xy_direct(self, Xtrain, ytrain, Xtest=None, ytest=None, Xval=None, yval=None):
        """
        Set X and y
        """

        self.size_sets = []

        self.Xtrain = Xtrain
        self.ytrain = ytrain            
        self.size_sets.append(len(ytrain))

        if yval is None:
            self.isVal = False
            self.size_sets.append(0)
        else:
            self.isVal = True
            self.Xval = Xval
            self.yval = yval
            self.size_sets.append(len(yval))
    
        if ytest is None:
            self.isTest = False
            self.size_sets.append(0)
        else:   
            self.isTest = True
            self.Xtest = Xtest
            self.ytest = ytest
            self.size_sets.append(len(ytest))

        self.size=np.sum(self.size_sets)
        print('\nSizes of train, validate, test:', self.size_sets, 'Total data:', self.size)


        
    def set_xy(self, target, features, ind=None):
        """
        Set X and y, make the train, validate, test split according to ind
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
        self.size=len(self.y)

        if ind is None:
            pass
        else:
            assert ind[0] <= len(self.y), \
                      "Error: ind[0] for max index for training data \
                       must be equal or smaller than len(target)" 
            assert ind[1] <= len(self.y), \
                      "Error: ind[1] for max index for validation data \
                       must be equal or smaller than len(target)" 

        self.size_sets = []
        self.isVal = False
        self.isTest = False
        
        if (ind is None or ind[0]==len(self.y)):
            # Just training data
            self.Xtrain = self.X[:len(self.y), :]
            self.ytrain = self.y[:len(self.y)]
            self.size_sets.append(len(self.ytrain))
        else:
            # We have more than just training data
            self.Xtrain = self.X[:ind[0], :]
            self.ytrain = self.y[:ind[0]]
            self.size_sets.append(len(self.ytrain))
            if (ind[1] > ind[0]):
                # Training, validation and test data
                self.isVal = True
                self.Xval   = self.X[ind[0]:ind[1], :]
                self.yval   = self.y[ind[0]:ind[1]]
                self.size_sets.append(len(self.yval))
                self.isTest = True
                self.Xtest  = self.X[ind[1]:, :]
                self.ytest  = self.y[ind[1]:]
                self.size_sets.append(len(self.ytest))
            else:
                # Training and test data
                self.size_sets.append(0)
                self.isTest = True
                self.Xtest  = self.X[ind[0]:, :]
                self.ytest  = self.y[ind[0]:]
                self.size_sets.append(len(self.ytest))

        
        if(self.debug is True):
            print('debug = TRUE')
            print('target:', self.y)
            print('ind, len y:', ind, len(self.y))
            print('ytrain:', self.ytrain)

        print('\nSizes of train, validate, test:', self.size_sets, 'Total data:', self.size)
        assert self.size == np.sum(self.size_sets), "Sum error in sizes of train, validate, test sets" 


                    
        
    def predict(self, X):
        X = np.array(X).reshape(1,-1)
        return self.model.predict(X)
        
        

