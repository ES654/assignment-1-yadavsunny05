
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

