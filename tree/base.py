"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion, max_depth = 4):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.tree = None
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        pass

    def fit(self, X, y):
        if(X.dtype[0] == "category" and Y.dtype[0] == "category"):
            for crit in self.criterion:
                if(crit == "information_gain"):
                    self.tree = ID3(self,X,Y,list(X.index),self.max_depth)
                elif(crit == "gini_index"):
                    self.tree = ID3_ginni(self,X,Y,list(X.index),self.max_depth) 
        elif(X.dtype[0] == "category" and Y.dtype[0] != "category"):

        elif(X.dtype[0]!="category" and Y.dtype[0] == "category"):
            
        elif(X.dtype[0] != "category" and Y.dtype[0] != "category"):


        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        pass
    
    def ID3(self,x,y,row_index,depth):
        tree = dict()
        subset_x = x.iloc[row_index,:]
        subset_y = y.iloc[row_index]
        if(len(list(set(subset_y))) == 1):
            tree['leaf']=y[0]
            return(tree)
        if(depth == 0):
            tree['end']=y
            return(tree)
        max_info = -99999
        info_key = ""
        for i in subset_x:
            temp = information_gain(subset_y,subset_x[i])
            if(temp>max_info):
                max_info = temp
                info_key = i
        if(info_key in row_index):
            if(len(row_index)>1):    
                row_index.remove(info_key)
        if(info_key!=""):
            tree[info_key] = dict()
            for i in set(subset_x[info_key]):
                tree[info_key][i]=ID3(self,x.drop(columns = info_key),y,list(subset_x[subset_x[info_key] == i].index),depth-1)
        return(tree)
    




    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pass

    def plot(self):
        """
        Function to plot the tree
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass