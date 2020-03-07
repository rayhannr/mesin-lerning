import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

#1 Loading iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

#Adding ones to the leftmost column of X.
X = np.append(np.ones((X.shape[0],1)),X,axis=1)

#1 Equally splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

#Indices of x and y are not 0 and 1 since the first column is rows of ones
x_index = 1
y_index = 2

#2 Visualizing training data
plt.figure(figsize=(6, 4))
plt.scatter(X_train[:, x_index], X_train[:, y_index], c=y_train)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()

#3 Sigmoid function
def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g

#4 Cost function, note: feature = X, target = y
def cost_function(theta, feature, target):
    hypothesis = sigmoid(np.dot(feature, theta))
    error = target * np.log(hypothesis) + (1 - target) * np.log(1 - hypothesis)
    cost = np.sum(error) / target.size * -1
    return cost

#Refactoring function to get new theta for every iteration in gradient_descent
def new_theta(feature, target, theta, alpha, index):
    hypothesis = sigmoid(np.dot(feature, theta))
    cf_derivative = (hypothesis - target) * feature[: ,index]
    return theta[index] - alpha * np.sum(cf_derivative) / target.size

#5 Gradient descent
def gradient_descent(feature, target, theta):
    J_history = []
    #6 Applying logistic regression model to the training data
    learning_rate = 0.001
    iterations = 100
    for i in range(0, iterations):
        temp0 = new_theta(feature, target, theta, learning_rate, 0)
        temp1 = new_theta(feature, target, theta, learning_rate, 1)
        temp2 = new_theta(feature, target, theta, learning_rate, 2)
        
        theta[0] = temp0
        theta[1] = temp1
        theta[2] = temp2
        J = cost_function(theta, feature, target)
        J_history.append(J)
        
    return theta, J_history

#Refactoring function to plot the decision boundary
def decision_boundary(ytrain, theta, title):
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train[: ,1:3], ytrain
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    contour = np.array([X1.ravel(), X2.ravel()]).T
    contour = np.append(np.ones((contour.shape[0], 1)), contour, axis=1)
    contour = sigmoid(np.dot(contour, theta))
    contour[contour < 0.5] = 0
    contour[contour >= 0.5] = 1
    contour = contour.reshape(X1.shape)

    plt.contourf(X1, X2, contour, alpha = 0.75, cmap = ListedColormap(('salmon', 'purple')))
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color = ListedColormap(('salmon', 'purple'))(i), label = j, s = 15)
    plt.title(title)
    plt.xlabel('sepal length(cm)')
    plt.ylabel('sepal width(cm)')
    plt.legend()
    plt.show()

#7 Training the model to learn class = 0. Class 0 will be marked as 1, otherwise will be 0
y0_train = y_train.copy()
y0_train[y0_train > 0] = 3
y0_train[y0_train == 0] = 1
y0_train[y0_train == 3] = 0
theta0, j0 = gradient_descent(X_train, y0_train, np.array([0.67, -2.4, 3.58]))
decision_boundary(y0_train, theta0, 'Decision Boundary for class 0')

#7 Training the model to learn class = 1. Class 1 will be marked as 1, otherwise will be 0
y1_train = y_train.copy()
y1_train[y1_train != 1] = 0
theta1, j1 = gradient_descent(X_train, y1_train,  np.array([4.75, 0.2, -1.9]))
decision_boundary(y1_train, theta1, 'Decision Boundary for class 1')

#7 Training the model to learn class = 2. Class 2 will be marked as 1, otherwise will be 0
y2_train = y_train.copy()
y2_train[y2_train == 1] = 0
y2_train[y2_train == 2] = 1
theta2, j2 = gradient_descent(X_train, y2_train, np.array([-2.3, 1.15, -1.5]))
decision_boundary(y2_train, theta2, 'Decision Boundary for class 2')

#Converting probability value to class label
def class_label(theta):
    probs = sigmoid(np.dot(X_test, theta))
    prediction = probs.copy()
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    
    return prediction

accuracy = []
cm = []

#Saving confusion matrix and accuracy of each class
from sklearn.metrics import confusion_matrix, accuracy_score
def acc_and_cm(prediction, test):
    cm.append(confusion_matrix(prediction, test))
    accuracy.append(accuracy_score(test, prediction))

#8 Applying the model to the test set with class = 0
prediction = class_label(theta0)
y0_test = y_test.copy()
y0_test[y0_test > 0] = 3
y0_test[y0_test == 0] = 1
y0_test[y0_test == 3] = 0
acc_and_cm(prediction, y0_test)

#8 Applying the model to the test set with class = 1
prediction = class_label(theta1)
y1_test = y_test.copy()
y1_test[y1_test != 1] = 0
acc_and_cm(prediction, y1_test)

#8 Applying the model to the test set with class = 2
prediction = class_label(theta2)
y2_test = y_test.copy()
y2_test[y2_test == 1] = 0
y2_test[y2_test == 2] = 1
acc_and_cm(prediction, y2_test)

total_accuracy = sum(accuracy) / 3

#9 Plotting cost function history
plt.plot(np.arange(100), j0, c='purple') #cost function to predict class = 0
plt.plot(np.arange(100), j1, c='pink') #cost function to predict class = 1
plt.plot(np.arange(100), j2, c='magenta') #cost function to predict class = 2
plt.xlabel('Iterations')
plt.ylabel('Cost function history')
plt.show()