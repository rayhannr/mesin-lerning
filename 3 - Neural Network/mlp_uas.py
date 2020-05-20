# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os

train_path = "./flowers/training_set/"
test_path = "./flowers/test_set/"

train_folders = os.listdir(train_path)
test_folders = os.listdir(test_path)

def preprocess_image(folders, path):
    # Import the images and resize them to a 128*128 size
    # Also generate the corresponding labels
    data_labels = []
    data_images = []
    size = 80, 60
    
    for folder in folders:
        for file in os.listdir(os.path.join(path, folder)):
            if file.endswith("jpg"):
                data_labels.append(folder)
                img = cv2.imread(os.path.join(path, folder, file))
                im = cv2.resize(img,size)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = im.flatten()
                im = im.astype('float32') / 255.0
                im = np.append([-1], im)
                data_images.append(im)
            else:
                continue
    
    #Transform the image array to a numpy type
    images = np.array(data_images)
    
    # Extract the labels
    label_dummies = pd.get_dummies(data_labels)
    labels = label_dummies.values.argmax(1)
    
    return images, labels

X_mlp_train, y_mlp_train = preprocess_image(train_folders, train_path)
X_test, y_mlp_test = preprocess_image(test_folders, test_path)

def one_hot_encoding(target):
    new_target = np.eye(np.max(target) + 1)[target]
    return new_target

y_mlp_train = one_hot_encoding(y_mlp_train)
y_mlp_test = one_hot_encoding(y_mlp_test)

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
    bias = -1 * np.ones((hidden_neuron.shape[0],1))
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

def set_delta(error, theta, neuron_val, target):
    delta = np.dot(error.T, neuron_val)
    return delta / len(target)

def update_weight(theta, delta):
    learning_rate = 0.3
    return theta - learning_rate * delta

def back_propagation(feature, target, hidden_neuron, output_neuron, theta1, theta2):
    output_error = (output_neuron - target) * gradient_sigmoid_function(output_neuron)
    hidden_error = get_hidden_layer_error(output_error, theta2, hidden_neuron)
    
    input_delta = set_delta(hidden_error, theta1, feature, target)
    hidden_delta = set_delta(output_error, theta2, hidden_neuron, target)
    
    new_theta1 = update_weight(theta1, input_delta)
    new_theta2 = update_weight(theta2, hidden_delta)
    
    return new_theta1, new_theta2

def calculate_accuracy(prediction, target):
    correct = 0
    for i in range(len(target)):
        if all(prediction[i] == target[i]):
            correct += 1
    return correct / float(len(target))

input_layer = 80 * 60
hidden_layer = 32
output_layer = 3
theta1, theta2 = initialize_network(input_layer, hidden_layer, output_layer)
hidden_neuron, output_neuron = forward_propagation(X_mlp_train, theta1, theta2)
train_error = []
train_accuracy = []
test_error = []
test_accuracy = []
epoch = 300

for i in range(epoch):
    hidden_neuron, output_neuron = forward_propagation(X_mlp_train, theta1, theta2)
    acc = calculate_accuracy(predict(output_neuron), y_mlp_train)
    cost = cost_function(output_neuron, y_mlp_train)
    train_accuracy.append(acc)
    train_error.append(cost)
    
    theta1, theta2 = back_propagation(X_mlp_train, y_mlp_train, hidden_neuron, output_neuron, theta1, theta2)
    
    hidden_neuron, output_neuron = forward_propagation(X_test, theta1, theta2)
    acc = calculate_accuracy(predict(output_neuron), y_mlp_test)
    cost = cost_function(output_neuron, y_mlp_test)
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

cok = predict(output_neuron)
if(all(cok[1] == y_mlp_train[1])):
    print('dick')