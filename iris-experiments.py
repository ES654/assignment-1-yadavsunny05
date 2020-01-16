import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

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

