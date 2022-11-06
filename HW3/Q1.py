import pandas as pd
import numpy as np
import math
import random as rd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import statistics


def PCA(data):
#Calculating Eigen values and vectors 
    values,vectors=np.linalg.eig(data.cov())
    # binding values and vector in a tuple
    eig_pairs = [(np.abs(values[i]), vectors[:,i]) for i in range(len(values))]
    #soring according to larger Eigen Values
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # Variance explaied 
    var_exp = [(i / sum(values))*100 for i in sorted(values, reverse=True)]
    variance_exp=np.cumsum(var_exp)
    return eig_pairs,variance_exp

def PCA_transform(data,eig_pairs,k):
    W=[]
    for i in range(k):
        W.append(eig_pairs[i][1])
        print(data[i][0])
    W=np.array(W)
    return np.dot(data,W.T)
    
data  = pd.read_csv("pima-indians-diabetes.csv",skiprows=9,header=None);
x = data.iloc[:,0:7];
y = data.iloc[:,-1];
    
scaler = StandardScaler();
x = pd.DataFrame(scaler.fit_transform(x));

eig_pair, var = PCA(x);
#print(var);
xtrans = PCA_transform(x,eig_pair,3);
#print(xtrans);
acc = [];
for i in range(10):
    x_train,x_test,y_train,y_test = train_test_split(xtrans,y,test_size=0.5);
    MLE =  GaussianNB();
    MLE.fit(x_train,y_train);
    acc.append(MLE.score(x_test,y_test));
print("Average Classification Accuracy: ",statistics.mean(acc));
