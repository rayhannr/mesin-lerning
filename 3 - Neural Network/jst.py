import numpy as np
from csv import reader
import matplotlib.pyplot as plt

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

dataset = load_csv('exam_data.csv')

def str_feature_to_float(dataset):
    X = list()
    for data in dataset:
        elements = list()
        elements.append(1)
        for el in data[0:7]:
            elements.append(float(el))
        X.append(elements)
        
    return np.array(X)

def str_target_to_integer(dataset):
    Y = list()
    for data in dataset:
        target = int(data[-1])
        Y.append(target)
    return np.array(Y)

X = str_feature_to_float(dataset)
Y = str_target_to_integer(dataset)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

def one_hot_encoding(target):
    new_target = np.eye(np.max(target) + 1)[target]
    return np.delete(new_target, obj=0, axis=1)

y_train = one_hot_encoding(y_train)
y_test = one_hot_encoding(y_test)

def initialize_network(n_inputs, n_hidden, n_outputs):
    theta1 = np.random.randn(n_hidden, n_inputs + 1)
    theta2 = np.random.randn(n_outputs, n_hidden + 1)
    
    return theta1, theta2

def sigmoid_function(z):
    z = z if np.isscalar(z) else np.asarray(z)
    sigmoid_value = 1.0 / (1.0 + np.exp(-z))
    return sigmoid_value

def gradient_sigmoid_function(z):
    z = z if np.isscalar(z) else np.asarray(z)
    grad_sigmoid = z * (1.0 - z)
    return grad_sigmoid

def cost_function(expected, target):
    cost = np.sum(np.square(expected - target)) / len(target)
    return cost

def get_neuron_value(theta, inputs):
    return sigmoid_function(np.dot(inputs, theta.T))

def forward_propagation(feature, theta1, theta2):
    hidden_neuron = get_neuron_value(theta1, feature)
    bias = np.ones((hidden_neuron.shape[0],1))
    hidden_neuron = np.append(bias, hidden_neuron, axis=1)
    output_neuron = get_neuron_value(theta2, hidden_neuron)
            
    return hidden_neuron, output_neuron

def predict(output):
    prediction = np.zeros(output.shape)
    for i in range(output.shape[0]):
        max_val = np.amax(output_neuron[i])
        for j in range(output.shape[1]):
            if(output[i][j] == max_val):
                prediction[i][j] = 1
            else:
                prediction[i][j] = 0
    return prediction
    
def get_hidden_layer_error(next_layer_error, theta, neuron_value):
    error = np.dot(next_layer_error, theta) * gradient_sigmoid_function(neuron_value)
    
    #delete error for bias neuron
    return np.delete(error, 0, axis=1)

def set_delta(error, neuron_val, target):
    delta = np.dot(error.T, neuron_val)
    return delta / len(target)

def update_weight(theta, delta):
    learning_rate = 0.3
    return theta - learning_rate * delta

def back_propagation(feature, target, hidden_neuron, output_neuron, theta1, theta2):
    output_error = (output_neuron - target) * gradient_sigmoid_function(output_neuron)
    hidden_error = get_hidden_layer_error(output_error, theta2, hidden_neuron)
    
    input_delta = set_delta(hidden_error, feature, target)
    hidden_delta = set_delta(output_error, hidden_neuron, target)
    
    new_theta1 = update_weight(theta1, input_delta)
    new_theta2 = update_weight(theta2, hidden_delta)
    
    return new_theta1, new_theta2

def calculate_accuracy(prediction, target):
    correct = 0
    for i in range(len(target)):
        if all(prediction[i] == target[i]):
            correct += 1
    return correct / float(len(target))

input_layer = 7
hidden_layer = 20
output_layer = 3
theta1, theta2 = initialize_network(input_layer, hidden_layer, output_layer)
hidden_neuron, output_neuron = forward_propagation(X_train, theta1, theta2)
train_error = []
train_accuracy = []
test_error = []
test_accuracy = []
epoch = 10000

for i in range(epoch):
    hidden_neuron, output_neuron = forward_propagation(X_train, theta1, theta2)
    acc = calculate_accuracy(predict(output_neuron), y_train)
    cost = cost_function(output_neuron, y_train)
    train_accuracy.append(acc)
    train_error.append(cost)
    
    theta1, theta2 = back_propagation(X_train, y_train, hidden_neuron, output_neuron, theta1, theta2)
    
    hidden_neuron, output_neuron = forward_propagation(X_test, theta1, theta2)
    acc = calculate_accuracy(predict(output_neuron), y_test)
    cost = cost_function(output_neuron, y_test)
    test_accuracy.append(acc)
    test_error.append(cost)
    
    
def visualize(train, test, ylabel, title):
    plt.plot(np.arange(len(train)), train, label='Train')
    plt.plot(np.arange(len(test)), test, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    
visualize(train_error, test_error, 'Error', 'Error')
visualize(train_accuracy, test_accuracy, 'Accuracy', 'Model Accuracy for Test Set')