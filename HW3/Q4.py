
import pandas as pd
import numpy as np
import random as rd
import statistics
from sklearn.metrics import accuracy_score

X=np.array([[1,1,-1,0,2],
[0,0,1,2,0],
[-1,-1,1,1,0],
[4,0,1,2,1],
[-1,1,1,1,0],
[-1,-1,-1,1,0],
[-1,1,1,2,1]]);

Y=np.array([2,1,2,1,1,1,2]);
start = np.array([3,1,1,-12,-7]);
f = np.dot(X,start);

step=1
w=start
for i in range(20):
    y_dash=0
    for i in range(0,len(X)):
        y_hat=[]
        f=np.dot(X[i],w)
        y_dash = 2 
        if (f>0):
            y_hat.append(y_dash);
        else:
            y_hat.append(1);
        for j in range(0, len(w)):  
            w[j] = w[j] + step*(Y[i]-y_dash)*X[i][j]
            
print("accuracy ",accuracy_score(y_hat,Y)*100);