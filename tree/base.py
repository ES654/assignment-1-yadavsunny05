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
from .utils import entropy, information_gain, gini_index,STD,best_split

np.random.seed(42)
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

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

    def fit(self, X, Y):
        if(X.dtype[0] == "category" and Y.dtype[0] == "category"):
            for crit in self.criterion:
                if(crit == "information_gain"):
                    self.tree = self.ID3(X,Y,list(X.index),self.max_depth)
                elif(crit == "gini_index"):
                    self.tree = None
        elif(X.dtype[0] == "category" and Y.dtype[0] != "category"):
            self.tree = self.ID3_discrete_real(X,Y,list(X.index),self.max_depth)  
        elif(X.dtype[0]!="category" and Y.dtype[0] == "category"):
            self.tree = self.ID3_discrete_real(X,Y,list(X.index),self.max_depth) 
        elif(X.dtype[0] != "category" and Y.dtype[0] != "category"):
            return  

        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """    
    def ID3(self,x,y,row_index,depth):
        tree = dict()
        subset_x = x.iloc[row_index,:]
        subset_y = y.iloc[row_index]
        if(len(list(set(subset_y))) == 1):
            tree['terminal']=y[0]
            return(tree)
        if(depth == 0):
            nums = dict()
            for i in y:
                if(i in nums):
                    nums[i]+=1
                else:
                    nums[i] = 1
            max_class = [key  for (key, value) in nums.items() if value == max(nums.values())][0]
            tree['terminal']= max_class
            return(tree)
        max_info = -99999
        info_key = ""
        temparr1 = []
        temparr2 = []
        for i in subset_x:
            temparr1.append(information_gain(subset_y,subset_x[i]))
            temparr2.append(i)
        max_info = max(temparr1)
        infor_key = temparr2[temparr1.index(max(temparr1))]
        if(info_key in row_index):
            if(len(row_index)>1):    
                row_index.remove(info_key)
        if(info_key!=""):
            tree[info_key] = dict()
            for i in set(subset_x[info_key]):
                tree[info_key][i]=self.ID3(x.drop(columns = info_key),y,list(subset_x[subset_x[info_key] == i].index),depth-1)
        return(tree)

    def ID3_discrete_real(self,x,y,row_index,depth):
        tree = dict()
        subset_x = x.iloc[row_index,:]
        subset_y = y.iloc[row_index]
        if(len(list(set(subset_y))) == 1):
            tree['terminal']=list(set(subset_y))[0]
            return(tree)
        if(depth == 0):
            tree['terminal']=np.mean(subset_y)
            print(np.mean(subset_y))
            return(tree)
        max_info = -99999
        info_key = ""
        temparr1 = []
        temparr2 = []
        for i in subset_x:
            temparr1.append(STD(subset_y,subset_x[i]))
            temparr2.append(i)
        max_info = max(temparr1)
        info_key = temparr2[temparr1.index(max(temparr1))]
        if(info_key in row_index):
            if(len(row_index)>1):    
                row_index.remove(info_key)
        if(info_key!=""):
            tree[info_key] = dict()
            for i in set(subset_x[info_key]):
                tree[info_key][i]=self.ID3_discrete_real(x.drop(columns = info_key),y,list(subset_x[subset_x[info_key] == i].index),depth-1)
        return(tree)

    def ID3_real_discrete(self,x,y,row_index,depth):
        node  = Node(None)
        subset_x = x.iloc[row_index,:]
        subset_y = y.iloc[row_index]
        if(depth == 0):
            nums = dict()
            for i in y:
                if(i in nums):
                    nums[i]+=1
                else:
                    nums[i] = 1
            max_class = [key  for (key, value) in nums.items() if value == max(nums.values())][0]
            node.predicted_class =  max_class
            return(node)
        if(len(list(set(subset_y))) == 1):
            node.predicted_class =list(set(subset_y))[0]
            return(node)
        else:
            max_info = 99999
            info_key = ""
            threshold=0
            for i in subset_x:
                split=best_split(subset_y,subset_x[i])
                if split[0]<max_info:
                    max_info=split[0]
                    threshold=split[1]
                    info_key=i
            node.feature_index = info_key
            node.threshold = split[1]
            node.right =self.ID3_real_discrete(x.drop(columns = info_key),y,list(subset_x[subset_x[info_key] >threshold ].index),depth-1)
            node.left =self.ID3_real_discrete(x.drop(columns = info_key),y,list(subset_x[subset_x[info_key] <=threshold].index),depth-1)
        return(node)




    def predict(self, X):
        if(X.dtype[0] != "category"):
            tree = self.tree
            if(type(tree) == type(dict)):
                return
            else:
                ans = []
                for i in range(len(X)):
                    tree1 = tree
                    while(tree1.predicted_class==None):
                        if(X.iloc[i][tree1.feature_index]> tree1.threshold):
                            tree1 = tree1.right
                        else:
                            tree1 = tree1.left
                    ans.append(tree1.predicted_class)
                return(ans)

        else:
            return






        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """

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