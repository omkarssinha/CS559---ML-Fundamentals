import numpy as np
import math


def printMeanAndVariance(data):
    mean =0;
    variance=0;
    for i in data:
        mean = mean + i;

    N = len(data);
    mean = mean/N;
    for i in data:
        variance = variance + math.pow((i - mean),2);

    variance = variance/N;
    print("Mean is ", mean ,"\nVariance is ", variance);  ## Print mean and variance of random set of data

def generateUserDefined(N, mean, var):    
    sd = math.sqrt(var);
    data = np.random.normal(mean,sd,N);
    print("\nGenerated Data of N obervations with user-defined mean and variance");
    print(data);
    ##printMeanAndVariance(data);   ## can check the mean and variance of the generated data
 

N = input("Enter N:");
mean = input("Enter mean:");
var = input("Enter variance:");
N = int(N);

data = np.random.randn(N);     ## Normal distribution with mean 0 and variance 1 obtained
print("\nGenerated data of N obervations with mean 0 and variance 1")
print(data);

data1 = np.random.rand(N);

print("\nMean and variable of a random dataset:");
printMeanAndVariance(data1);

generateUserDefined(N, float(mean), float(var));

dataset1 = np.random.normal(1,2,size=(2000,1));
dataset2 = np.random.normal(4,3,size=(1000,1));

print("\nMean and variable of a combined dataset(Q3 part 2):");
dataset3 = np.concatenate((dataset1,dataset2));
printMeanAndVariance(dataset3);








