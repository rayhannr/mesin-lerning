import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data1.txt', sep=',', header=None)
X = dataset[0]
y = dataset[1]
m = y.size

plt.scatter(X, y, color = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

X = np.stack([np.ones(m), X], axis=1)

def costFunction(x, y, theta):
    J = 0
    J = (1/(2 * m)) * np.sum(np.square(np.dot(x, theta) - y)) # compute cost using the above cost function
    return J

def gradientDescent(feature, result, theta, alpha, num_iters):
    costFunctions = []
    thetas0 = []
    thetas1 = []
    for i in range(0, num_iters):
        temp0 = theta[0] - alpha * (1 / result.size) * np.sum(np.dot(feature, theta) - result)
        temp1 = theta[1] - alpha * (1 / result.size) * np.sum((np.dot(feature, theta) - result) * feature[:, 1])
        theta[0] = temp0
        theta[1] = temp1
        J = costFunction(feature, result, theta)
        thetas0.append(theta[0])
        thetas1.append(theta[1])
        costFunctions.append(J)
        
    return thetas0, thetas1, costFunctions
    
learningRate = 0.001
iterations = 10000
thetas0, thetas1, J_history = gradientDescent(X, y, np.zeros(2), learningRate, iterations)

#Plotting cost function history
plt.plot(np.arange(iterations), J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost function history')
plt.show()

#Plotting theta's transformation per iteration
plt.plot(thetas0, np.arange(iterations), color = 'blue')
plt.plot(thetas1, np.arange(iterations), color = 'red')
plt.xlabel('Thetas')
plt.ylabel('Iterations')
plt.show()

#Plotting the regression model
plt.scatter(X[:, 1], y, c='black', s=25)
plt.plot(X[:, 1], thetas0[-1] + thetas1[-1] * X[:, 1], c='blue')
plt.show()

#Plotting the connection between theta and cost function
plt.plot(thetas0, J_history)
plt.plot(thetas1, J_history)
plt.xlabel('Thetas')
plt.ylabel('Cost function')
plt.show()