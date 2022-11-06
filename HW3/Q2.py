
import pandas as pd
import numpy as np
import random as rd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

d1 = np.array([[-2,1], [-5,-4], [-3,1], [0,-3], [-8,-1]])
y1 = [1,1,1,1,1];
d2 = np.array([[2,5], [1,0], [5,-1], [-1,-3], [6,1]])
y2 = [2,2,2,2,2];
n1 = 5;
n2 = 5;
x = np.array([[-2,1], [-5,-4], [-3,1], [0,-3], [-8,-1], [2,5], [1,0], [5,-1], [-1,-3], [6,1] ])
y = y1+y2;

mu1=np.mean(d1.T,axis=1).reshape((2,1))
mu2=np.mean(d2.T,axis=1).reshape((2,1))
mu=np.mean((mu1+mu2),axis=1).reshape((2,1))

withinClass = np.cov(d1.T)+np.cov(d2.T)
print("within class variance  ",withinClass)

betweenClass = 5*((mu1-mu)*(mu1-mu).T)+5*((mu2-mu)*(mu2-mu).T);

print("between class variance ",betweenClass)

lda().fit(x, y);