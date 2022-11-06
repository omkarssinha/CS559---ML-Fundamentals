import numpy as np
import math
import random
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.optimize as opt # minimizing procedure
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv("pima-indians-diabetes.csv", header=None, engine="python", comment='#', usecols=[1,2,3,8]);

col1_mean = df.iloc[0].mean(skipna=True);
df.iloc[0] = df.iloc[0].mask(df.iloc[0] == 0,col1_mean);
#col2_mean = df.iloc[1].mean(skipna=True);
#df.iloc[1] = df.iloc[1].mask(df.iloc[1] == 0,col2_mean);
col3_mean = df.iloc[3].mean(skipna=True);
df.iloc[2] = df.iloc[2].mask(df.iloc[2] == 0,col3_mean);
print(col3_mean)
#df.replace(0,df.mean(axis=0),inplace=True)
#df = df.mask(df==0).fillna(df.mean(skipna=True))
##df.set_index('ID', inplace=True) # set ID as index

#Data = df['pima-indians-diabetes.csv']                                     # Load data
#Data = Data.dropna()
##Data = np.log(Data)                                                      # Take log() transformation

#print(pd.DataFrame.equals(Data));

##print(df)
"""
with open("pima-indians-diabetes.csv", "r") as f:
    data = f.read().split('\n')

random.shuffle(data)

train_data = data[:384]
test_data = data[384:]

print(train_data);"""


class MLClassifier:
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n,)
        '''
        # no. of variables / dimension
        self.d = x.shape[1]
        
        # no. of classes; assumes labels to be integers from 0 to nclasses-1
        self.nclasses = len(set(y))
        
        # list of means; mu_list[i] is mean vector for label i
        self.mu_list = []
        
        # list of inverse covariance matrices;
        # sigma_list[i] is inverse covariance matrix for label i
        # for efficiency reasons we store only the inverses
        self.sigma_inv_list = []
        
        # list of scalars in front of e^...
        self.scalars = []
        
        n = x.shape[0]
        for i in range(self.nclasses):
            
            # subset of obesrvations for label i
            cls_x = np.array([x[j] for j in range(n) if y[j] == i])
            
            mu = np.mean(cls_x, axis=0)
            
            # rowvar = False, this is to use columns as variables instead of rows
            sigma = np.cov(cls_x, rowvar=False)
            if np.sum(np.linalg.eigvals(sigma) <= 0) != 0:
                # if at least one eigenvalue is <= 0 show warning
                print(f'Warning! Covariance matrix for label {cls} is not positive definite!\n')
            
            sigma_inv = np.linalg.inv(sigma)
            
            scalar = 1/np.sqrt(((2*np.pi)**self.d)*np.linalg.det(sigma))
            
            self.mu_list.append(mu)
            self.sigma_inv_list.append(sigma_inv)
            self.scalars.append(scalar)
    
    def _class_likelihood(self, x: np.ndarray, cls: int) -> float:
        '''
        x - numpy array of shape (d,)
        cls - class label
        
        Returns: likelihood of x under the assumption that class label is cls
        '''
        mu = self.mu_list[cls]
        sigma_inv = self.sigma_inv_list[cls]
        scalar = self.scalars[cls]
        d = self.d
        
        exp = (-1/2)*np.dot(np.matmul(x-mu, sigma_inv), x-mu)
        
        return scalar * (np.e**exp)
    
    def predict(self, x: np.ndarray) -> int:
        '''
        x - numpy array of shape (d,)
        Returns: predicted label
        '''
        likelihoods = [self._class_likelihood(x, i) for i in range(self.nclasses)]
        return np.argmax(likelihoods)
    
    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n,)
        Returns: accuracy of predictions
        '''
        n = x.shape[0]
        predicted_y = np.array([self.predict(x[i]) for i in range(n)])
        n_correct = np.sum(predicted_y == y)
        return n_correct/n

accuracyList =[]
mean=0;
stdev=0;
n= 10;

for x in range(n):

    (x_train, x_test, y_train, y_test) = train_test_split(
    df.iloc[:, 0:3].values, df.iloc[:, 3].values, train_size=0.5);

    mlc = MLClassifier()
    mlc.fit(x_train, y_train)

    score = mlc.score(x_test, y_test);

    print('Test score: ')
    print(score)
    accuracyList.append(score)

for i in accuracyList:
    mean = mean + i

mean = mean/n;
stdev = np.std(accuracyList, ddof=1)

print('\nAccuracy mean: ')
print(mean)
print('\nAccuracy standard deviation: ')
print(stdev)