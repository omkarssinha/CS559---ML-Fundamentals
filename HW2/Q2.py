import numpy as np
import math
import random
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from numpy.random import randint

def eucledian_dist(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

def predict(x_train, y , x_input, k):
    op_labels = []
    for item in x_input:         
        point_dist = []
        for j in range(len(x_train)): 
            distances = eucledian_dist(np.array(x_train[j,:]) , item) 
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
        dist = np.argsort(point_dist)[:k]          
        labels = y[dist]
        n = mode(labels) 
        n = n.mode[0]
        op_labels.append(n)
 
    return op_labels

df = pd.read_csv("pima-indians-diabetes.csv", header=None, engine="python", comment='#')

k_list = [1,5,11]
mean_acc = []
sd_acc = []

for k in k_list:
    accuracies = []
     
    for i in range(10):    
        X = df.iloc[:,1:4]
        y = df.iloc[:,8]
        
        (x_train, x_test, y_train, y_test) = train_test_split(X.values, y.values, train_size=0.5)
        
        y_pred = predict(x_train,y_train,x_test, k)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    mean_acc.append(np.mean(accuracies)*100)
    sd_acc.append(np.std(accuracies))

data = {'k': k_list, 'Mean of Accuracy': mean_acc, 'Standard Deviation of Accuracy': sd_acc}

result  = pd.DataFrame(data)
print(result)
