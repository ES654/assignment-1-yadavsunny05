
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn import tree as tree5

np.random.seed(42)

# Read real-estate data set
# ...
# 
data = pd.read_csv(r'C:\Users\Anshuman Yadav\Documents\Real.csv')
X_train,X_test,Y_train,Y_test = train_test_split(data[data.columns[1:-1]],data[data.columns[-1]])
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)
X_train.dtype = "d"
X_test.dtype = "d"



tree = DecisionTree("ad",max_depth = 25)
tree.fit(X_train,Y_train)
tree.root
y_pred = tree.predict(X_test)
print("MAE my tree : -")
print(mae(np.array(Y_test),np.array(y_pred)))
print("MSE my tree : -")
print(rmse(np.array(Y_test),np.array(y_pred)))

d_tree_sklearn = tree5.DecisionTreeRegressor()
d_tree_sklearn = d_tree_sklearn.fit(X_train,Y_train)
y_sklearn = d_tree_sklearn.predict(X_test)
print("MAE sklearn : -")
print(mae(np.array(Y_test),np.array(y_sklearn)))
print("MSE sklearn : -")
print(rmse(np.array(Y_test),np.array(y_sklearn)))

