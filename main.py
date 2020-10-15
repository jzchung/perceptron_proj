#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
'''
This is a simple program for perceptron. Author by Jessie Chung
Perceptron is a classic two-classification algorithm, which aims to learn a hyperplan: w^Tx + b = 0.
w is the weight vector and b is the bias. The objective function is defined as follows:
min_{w,b} L(w,b) = - \sum_{x_i \in M} y_i(wx_i+b)
where M is the miss-classified samples. The objective means to minimize the sum of distances from the miss-classified samples to the hyperplane.
Gradient descend is used to optimize the objective function. The main steps of the algorithm comes as:
========================
input: training dataset T={(x_i , y_i)}, learning rate eta \in (0,1]
output: w,b, that is, f(x)=sign(wx+b)
1. initialize w_0, b_0;
2. for (xi, yi) in T
3. if yi(wxi + b) \leq 0
    w <- w + eta*yi*xi
    b <- b + eta*yi      //check the gradient descend
4. end for
========================
'''


def percep(trainx, trainy, dim):
    b = 0
    eta = 1 # learning rate
    missflag = True # 0--no miss-classified 1--exists miss-classified
    w = np.zeros([1,dim]) #dtype=double default
    while(True):
        if missflag:
            print('exits miss-classified samples...')
            missflag = False
            for x,y in zip(trainx.itertuples(index=False),trainy):
                xp = np.array(x)
                if y*(np.dot(w,xp)+b) <= 0:
                    print('sample ',xp, 'miss-classified.', end='')
                    w += eta * y * xp
                    b += eta * y
                    missflag = True;
                    print('update w=' , w,' b=', b)
                else:
                        print('sample ', xp, 'successfully classified')
        else:
            print('finished')
            break
    return w,b


#load data
filename = './data/test.csv';
data = pd.read_csv(filename, header=None, sep=',') # read csv into dataframe
labels = data.iloc[-1]
data = data.drop(data.columns[len(data.columns)-1], axis = 1)
length = len(data.columns)
[w,b]=percep(data, labels, length)
print(w,b)
