# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:06:34 2019
@author: DELL
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
url = "D:/CS/DL_Tutorial/L1"
data = pd.read_csv(url + '/data_linear.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')
x = np.hstack((np.ones((N, 1)), x))
w = np.array([0.,1.]).reshape(-1,1)
numOfIteration = 100
cost = np.zeros((numOfIteration,1))
learning_rate = 0.000001
print(x)
# update w in linear regression
cost = []
for i in range(1, numOfIteration):
    r = np.dot(x, w) - y       
    cost[i] = 0.5*np.sum(r*r) # 1/2 ||r||^2 
    # Note that J_w(1/2 ||xw - y||^2) = np.dot(x.T,(xw - y)) 
    # --> we update coefficient vector using gradient descent
    w[0] -= learning_rate*np.sum(r) 
    w[1] -= learning_rate*np.sum(np.multiply(r, x[:,1].reshape(-1,1)))
    cost.append(cost[i])
predict = np.dot(x, w)
plt.plot((x[0][1], x[N-1][1]),(predict[0], predict[N-1]), 'r')
plt.show()

x1 = 50
y1 = w[0] + w[1] * 50
print('Giá nhà cho 50m^2 là : ', y1)