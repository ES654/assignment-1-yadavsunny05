
import math
import numpy as np

def entropy(Y):
    sample_count,num =  np.unique(Y,return_counts=True)
    entro = 0.0
    for i in num:
        if(i!=0):    
            prob = (i/np.sum(num))
            entro -= (prob*math.log(prob,2))
    return entro
    pass

    """
    Function to calculate the entropy 
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """

    """
    Function to calculate the entropy 
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """

    pass

def split(rows,threshold,col):
    left,right = [],[]
    for row in rows:
        if(row[col]>=threshold):
            right.append(row)
        else:
            left.append(row)
    return left,right

def info_gain_real_discrete(left,right,current):
    return entropy(current) - entropy(left[:-1])*len(left)/(len(left)+ len(right)) - entropy(right[:-1])*len(right)/(len(left)+len(right))


def best_split(rows):
    best_gain=-9999
    threshold= -9999
    index = 0
    best_left = None
    best_right = None
    current=entropy(rows[:,-1])
    for col in range(len(rows[0])-1):
        thresholds=set([row[col] for row in rows])
        for i in thresholds: 
            left,right=split(rows,i,col)
            if (left!=None and right !=None):  
                temp=info_gain_real_discrete(left,right,rows[:,-1])
                if temp>=best_gain: 
                    best_left = np.array(left)
                    best_right  = np.array(right)
                    best_gain = temp
                    threshold=i
                    index = col
    return best_gain,threshold,index,best_left,best_right

def gini_index(Y):
    sample_count =  Y.iloc[:, -1].value_counts()
    num = Y.shape[0]
    gini = 1.0
    for i in sample_count.values():
        prob = (i/num)
        gini -= (prob*prob)
    return gini
    pass

    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """

def gini_info(Y,attr):
    initial_gini = gini_index(Y[attr])
    gini_val = 0.0



def STD(Y, attr):
    initial_variance = np.var(np.array(Y))
    y_val = dict()
    indexes=list(Y.index)
    for index in indexes:
            y_val[attr[index]]=[]
    for index in indexes:
        y_val[attr[index]].append(Y[index])
    for i in y_val:
        initial_variance-=np.var(np.array(y_val[i]))*(len(y_val[i])/len(list(Y.index)))
    return(initial_variance)

def best_split_Real_Real(Y,attr):
    splitarr = []
    for i in range(len(attr)):
        splitarr.append([Y[i],attr[i]])
    splitarr = np.array(sorted(splitarr,key = lambda x:x[0]))
    min_info=999999
    split_value=0
    for i in range(1,len(Y)):
        temp = np.var(splitarr[:i,1]) + np.var(splitarr[i:,1])
        if(min_info>temp):
            min_info = temp
            split_value=(splitarr[i,1]+splitarr[i-1,1])/2 
    return((min_info,split_value))


def information_gain(Y, attr):
    initial_gain = entropy(Y)
    y_val = dict()
    indexes=list(Y.index)
    for index in indexes:
            y_val[attr[index]]=[]
    for index in indexes:
        y_val[attr[index]].append(Y[index])
    for i in y_val:
        initial_gain-=entropy(y_val[i])*(len(y_val[i])/len(list(Y.index)))
    return(initial_gain)


    """
    Function     to calculate the information gain
    Inputs:
    > Y: pd.Series of Labels
    > attr: attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    pass


    """
    Function     to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """

