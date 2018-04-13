import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(x, theta):
    z = np.transpose(theta).dot(x)
    ret = sigmoid(z)
    if (ret >= 0.5):
        return 1
    else:
        return 0

def cost(x, y, theta):
    m = len(x)
    _sum = 0
    for i in range(m):
        _sum += (y[i] * np.log(sigmoid(np.dot(x[i], theta)))) + ((1 - y[i]) * np.log(1 - sigmoid(np.dot(x[i], theta))))
    return -(1 / m) * _sum

def scaling_x(x):
    tab1 = []
    tab2 = []
    j = 0;
    for i in x:
        tab1.append(i[1])
        tab2.append(i[2])
    for i in x:
        i[1] = i[1] / (np.amax(tab1) - np.amin(tab1))
        i[2] = i[2] / (np.amax(tab2) - np.amin(tab2))
        x[j] = i
        j += 1
    return x

def scaling_theta(theta, x):
    tab1 = []
    tab2 = []
    for i in x:
        tab1.append(i[1])
        tab2.append(i[2])
    theta[1] = theta[1] / (np.amax(tab1) - np.amin(tab1))
    theta[2] = theta[2] / (np.amax(tab2) - np.amin(tab2))
    return theta


def training(m, x, y, j, theta):
    _sum = 0
    for i in range(m):
        _sum += (sigmoid(np.dot(x[i], theta)) - y[i]) * x[i][j]
    return _sum

def gradient_descent(x, y, theta, alpha):
    m = len(y)
    _cost = 0
    while _cost != cost(x, y, theta):
        _cost = cost(x, y, theta)
        for i in range(len(theta)):
            theta[i] = theta[i] - (alpha / m) * training(m, x, y, i, theta)
    return theta

data = pd.read_csv("ex2data1.csv")
x = [np.insert(row, 0, 1) for row in np.array(data,float)[:, :2]]
y = np.array(data,float)[:, 2]

theta = np.array([[0], [0], [0]],float)

alpha = 0.9
old_x = np.copy(x)
x = scaling_x(x)
theta = gradient_descent(x, y, theta, alpha)
theta = scaling_theta(theta, old_x)
print("theta : {0}".format(theta))
print("cost : {0}".format(cost(x, y, theta)))
print("prediction : {0}".format(predict([1, 40, 90], theta)))
