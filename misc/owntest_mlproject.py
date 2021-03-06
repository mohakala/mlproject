# -*- coding: utf-8 -*-
"""
Own test of the class
"""


import numpy as np
#import pandas as pd

import sys
sys.path.insert(0, 'C:\Python34')

from mlproject import mlproject as mlp


def addFeat(df):
    """
    In this function we add new features to the data table
    We encode the SMILES formula to some values
    These new features are added as new columns to the table
    - number of C, O, N in the formula
    - relative values: n[O]/n[C], n[N]/n[C], (n[N]+n[O])/n[C] in the formula
    You can also make yourself new features following the examples below
    """
    # Calculate how many elements there are in the formula
    # and insert this value into new columns
    # Note: you can also make yourself new columns
    
    ind=0
    for formula in df['SMILES']:
        no=formula.count('O')
        nn=formula.count('N')
        nc=formula.count('C') + formula.count('c')
        ndouble=formula.count('=')
        ncl=formula.count('Cl')
        # 'n_o': number of oxygens in the formula
        df.set_value(ind,'n_o', no)
        df.set_value(ind,'n_n', nn)
        df.set_value(ind,'n_c', nc)
        df.set_value(ind,'n_cl', ncl)
        df.set_value(ind,'n_dbl', ndouble)    
        # 'n_o_c': relative number of oxygens with respect to carbons
        df.set_value(ind,'n_o_c', no/nc)
        df.set_value(ind,'n_n_c', nn/nc)
        df.set_value(ind,'n_no_c', (nn+no)/nc)
        df.set_value(ind,'n_dbl_c', ndouble/nc)
        df.set_value(ind,'n_cl_c', ndouble/nc)
        ind+=1
    return(df)


ml = mlp.mlproject()
path = 'C:\\Python34\datasets\solubility.txt'
ml.getData(path)
ml.examine()

ml.addHeader('Project 1')
ml.modifyData(addFeat)
#ml.head(3)

# Shorten the column name of the solubilities: --> sol and sol_pred
columns = {'measured log(solubility:mol/L)': 'sol', 'ESOL predicted log(solubility:mol/L)': 'sol_pred'}
ml.rename(columns)
ml.head(3)
# ml.plot(ml.df['n_c'], ml.df['sol'])

# print(ml.df.corr('pearson'))


# Choose X and y
target = 'sol'
#features= ['n_c', 'n_dbl']
features= ['n_o','n_n','n_c','n_cl','n_dbl','n_o_c','n_n_c','n_no_c','n_dbl_c','n_cl_c']

# Drop non-numerical columns
ml.dropColumn('Compound ID')
ml.dropColumn('SMILES')

# Normalize the values in df 
# Subtract the mean and divide by the data range
ml.normalize()

ml.randomizeRows()

# Version 1: give indices for train, val, test 
ind = [800, 1000]
#ml.set_xy(target, features, ind)


# Version 2: use train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(ml.df[features].values, ml.df[target].values, test_size=0.2)
ml.set_xy_direct(X_train, y_train, X_test, y_test)


print('*KNN')
from sklearn.neighbors import KNeighborsRegressor

# Select hyperparameters for KNN
weights = 'uniform'     #  'uniform' or 'distance'
if(False):
    for n in range(1, 12):
        model = KNeighborsRegressor(n_neighbors=n, weights=weights)
        print('param, score:', n, ml.score(model, iprint=0))
weights = 'uniform'     #  'uniform' or 'distance'
n_neighbors = 6
model = KNeighborsRegressor(n_neighbors, weights=weights)
ml.score(model, printTestScore=True)


from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor(KNeighborsRegressor(n_neighbors = 6, weights='uniform'),  \
                         n_estimators=20, max_samples=1.0)
ml.score(model, printTestScore=True)


print('*Decision tree')
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
ml.score(model)

from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor(DecisionTreeRegressor(), n_estimators=15, max_samples=1.0)
ml.score(model)

from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=10)
ml.score(model, iprint=3)


print('*RF:')
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
ml.score(model, printTestScore=True)

#for i in range(50 , 500, 100):
#    model=RandomForestRegressor(n_estimators=i)
#    print('i:', i, ml.score(model, iprint=0))



print('*Linear, ridge, lasso and kernel ridge regression')

from sklearn.linear_model import LinearRegression
model = LinearRegression()
ml.score(model)
#print(model.coef_)


from sklearn.linear_model import Ridge
# Select the hyperparameter for ridge regression
if(False):
    for alpha in np.linspace(0.01, 3, 30):
        model = Ridge(alpha=alpha)
        print('alpha, score:', alpha, ml.score(model, iprint=0))
print('**ridge: selected alpha, but no improvement = ', 0.3, ':')
alpha = 0.3
model = Ridge(alpha=alpha)
ml.score(model)
#print(model.coef_)


from sklearn.linear_model import Lasso
# Select the hyperparameter for lasso regression
if(False):
    for alpha in np.linspace(0.0001, 0.1, 40):
        model = Lasso(alpha=alpha)
        print('alpha, score:', alpha, ml.score(model, iprint=0))
print('**lasso: alpha very small, so no improvement. Use = 0.0001:')
alpha=0.0001
model = Lasso(alpha=alpha)
ml.score(model)
#print(model.coef_)




print('**linear kernel')
from sklearn.kernel_ridge import KernelRidge
model=KernelRidge(alpha=0.3)
ml.score(model)


# http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge
kernel='rbf'
from sklearn.kernel_ridge import KernelRidge


# Grid search for alpha and gamma hyperparameters
# Note, in KRR kernel: alpha = (2*C)^-1  
if(False):
    C_range = np.logspace(-2, 7, 10)
    alpha_range = 1.0 / (2*C_range)
    gamma_range = np.logspace(-7, 2, 10)
    # gamma_range = np.linspace(1, 30, 30)
    for gamma in gamma_range:
        for alpha in alpha_range: 
            model = KernelRidge(kernel=kernel, gamma=gamma, alpha=alpha)
            print('alpha, gamma, score:', alpha, gamma, ml.score(model, iprint=0))

print('**rbf: select alpha=0.05, gamma = 1:')
model = KernelRidge(kernel=kernel, gamma=1.0, alpha=0.05)
ml.score(model, iprint=4)    
    


print('------------')
print('Done')



