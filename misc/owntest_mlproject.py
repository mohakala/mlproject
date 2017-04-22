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
ml.plot(ml.df['n_c'], ml.df['sol'])

# print(ml.df.corr('pearson'))


# Choose X and y
target = 'sol'
#features= ['n_c', 'n_dbl']
features= ['n_o','n_n','n_c','n_cl','n_dbl','n_o_c','n_n_c','n_no_c','n_dbl_c','n_cl_c']

ind = [800, 1000]
ml.randomizeRows()
ml.set_new(target, features, ind)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
ml.score(model)


from sklearn import neighbors
weights = 'uniform'     #  'uniform' or 'distance'
n_neighbors = 6
model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
ml.score(model, iprint=True)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
ml.score(model)

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
ml.score(model, printTestScore=True)

print('------------')
print('Done')



