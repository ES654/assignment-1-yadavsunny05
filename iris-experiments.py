import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn import tree as tree5

np.random.seed(42)

# Read IRIS data set
# ...
# 
from sklearn.datasets import  load_iris
data=pd.DataFrame(load_iris()['data'])
n=len(data)
y = pd.DataFrame(load_iris()["target"])

X_train,X_test,Y_train,Y_test = train_test_split(data,y)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)

X_train.dtype = "da"
Y_train.dtype = "category"
X_test.dtype = "d"
X_train1 = X_train.copy()
X_train1.dtype = "d"

tree = DecisionTree("information_gain",max_depth = 6)
tree.fit(X_train1,Y_train)
tree.root
y_pred = tree.predict(X_test)
print(accuracy(np.array(Y_test),np.array(y_pred)))




d_tree_sklearn = tree5.DecisionTreeRegressor()
d_tree_sklearn = d_tree_sklearn.fit(X_train,Y_train)
y_sklearn = d_tree_sklearn.predict(X_test)
print(accuracy(np.array(y_sklearn),np.array(Y_test)))


npy = np.array(Y_train)
classes = set()
for i in range(len(npy)):
    classes.add(npy[i][0])
    
for i in classes:
    print("Precision My Tree for Class" + str(i) )
    print(precision(y_pred,Y_test,i))
    print("Recall My Tree for Class" + str(i))
    print(recall(y_pred,Y_test,i))
    print("Precision Sklearn for Class" + str(i))
    print(precision(y_sklearn,Y_test,i))
    print("Recall Sklearn for Class" + str(i))
    print(recall(y_sklearn,Y_test,i))



def k_fold(data,y, k =5):
    acc = []
    for i in range(5):
        val = len(data)//k
        x_test = data[val*i:val*(i+1)]
        x_train = np.append(data[0:val*i],data[val*(i+1):],axis = 0)
        y_test = y[val*i:val*(i+1)]
        y_train = np.append(y[0:val*i],y[val*(i+1):],axis = 0)
        x_train = pd.DataFrame(x_train)
        x_train.dtype = "dfs"
        x_test = pd.DataFrame(x_test)
        x_test.dtype = "sda"
        y_train = pd.DataFrame(y_train)
        y_train.dtype = "category"
        y_test = pd.DataFrame(y_test)
        y_test.dtype = "category"
        tree.fit(x_train,y_train)
        acc.append(accuracy(np.array(Y_test),np.array(tree.predict(X_test))))
        print(acc[i])
    print("Average : -")
    print(np.mean(acc))
    
k_fold(data,y,5)

def nested_cross(data,y,k1 = 5, k2 = 4):
    val1 = len(data)//k1
    for i in range(k1):
        y_test = y[val1*i:val1*(i+1)]
        x_test = data[val1*i:val1*(i+1)]
        x_train = np.append(data[0:val1*i],data[val1*(i+1):],axis = 0)
        y_train = np.append(y[0:val1*i],y[val1*(i+1):],axis = 0)
        acc = []
        for depth in range(2,10):
            s = 0
            for j in range(4):
                val2 = len(x_train)//k2
                x_val_test = x_train[val2*j:val2*(j+1)]
                y_val_test = y_train[val2*j:val2*(j+1)]
                x_val_train = np.append(x_train[0:val2*j],x_train[val2*(j+1):],axis = 0)
                y_val_train = np.append(y_train[0:val2*j],y_train[val2*(j+1):],axis = 0)
                tree = DecisionTree("information_gain",max_depth = depth)
                x_val_train = pd.DataFrame(x_val_train)
                y_val_train = pd.DataFrame(y_val_train)
                x_val_test = pd.DataFrame(x_val_test)
                y_val_test = pd.DataFrame(y_val_test)
                x_val_train.dtype = "sda"
                y_val_train.dtype = "category"
                x_val_test.dtype = "sda"
                y_val_test.dtype = "category"
                tree.fit(x_val_train,y_val_train)
                s+=(accuracy(np.array(y_val_test),np.array(tree.predict(x_val_test))))
            acc.append(s/4)
        value = max(acc)
        index = acc.index(max(acc))
        tree = DecisionTree("information_gain",max_depth =value)
        print("Best Accuracy is : - " + str(value))
        print("At Depth : - " + str(index+1))

nested_cross(data,y)
        
    





















