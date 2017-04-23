# -*- coding: utf-8 -*-
"""
Own test of the class
"""


#import numpy as np
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
ml.head(3)

# Shorten the column name of the solubilities: --> sol and sol_pred
columns = {'measured log(solubility:mol/L)': 'sol', 'ESOL predicted log(solubility:mol/L)': 'sol_pred'}
ml.rename(columns)
# ml.head()
# ml.plot(ml.df['n_c'], ml.df['sol'])

# print(ml.df.corr('pearson'))


# Choose X and y
target = 'sol'
#features= ['n_c', 'n_dbl']
features= ['n_o','n_n','n_c','n_cl','n_dbl','n_o_c','n_n_c','n_no_c','n_dbl_c','n_cl_c']


ml.randomizeRows()

# Version 1: give indices for train, val, test 
ind = [800, 1000]
#ml.set_xy(target, features, ind)


# Version 2: use train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(ml.df[features].values, ml.df[target].values, test_size=0.2)
ml.set_xy_direct(X_train, y_train, X_test, y_test)


print('KNN')
from sklearn.neighbors import KNeighborsRegressor
weights = 'uniform'     #  'uniform' or 'distance'
n_neighbors = 6
model = KNeighborsRegressor(n_neighbors, weights=weights)
ml.score(model)

from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor(KNeighborsRegressor(n_neighbors = 6), n_estimators=10, max_samples=1.0)
ml.score(model)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
ml.score(model)

from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor(DecisionTreeRegressor(), n_estimators=15, max_samples=1.0)
print('cv, bagging, dec.tree:')
ml.score(model)

from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=10)
print('cv, adaboost, dec.tree:')
ml.score(model, iprint=3)

print('RF:')
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
ml.score(model, printTestScore=True)


#for i in range(50 , 500, 100):
#    model=RandomForestRegressor(n_estimators=i)
#    print('i:', i, ml.score(model, iprint=0))


print('Linear, ridge and kernel ridge')

from sklearn.linear_model import LinearRegression
model = LinearRegression()
ml.score(model)
print(model.coef_)


from sklearn.linear_model import Ridge

import numpy as np
alpha = np.linspace(0.01, 8, 10)
for i in alpha:
    model = Ridge(alpha=i)
    print('alpha, score:', i, ml.score(model, iprint=0))


print('- linear kernel')
from sklearn.kernel_ridge import KernelRidge
model=KernelRidge()
ml.score(model)

    
    
print('------------')
print('Done')



