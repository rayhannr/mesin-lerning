# Import the necessary libraries
from tensorflow import keras
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
    size = 160, 120
    
    for folder in folders:
        for file in os.listdir(os.path.join(path, folder)):
            if file.endswith("jpg"):
                data_labels.append(folder)
                img = cv2.imread(os.path.join(path, folder, file))
                im = cv2.resize(img,size)
                data_images.append(im)
            else:
                continue
    
    #Transform the image array to a numpy type
    images = np.array(data_images)
    
    # Reduce the RGB values between 0 and 1
    images = images.astype('float32') / 255.0
    
    # Extract the labels
    label_dummies = pd.get_dummies(data_labels)
    labels =  label_dummies.values.argmax(1)
    
    return images, labels

X_train, y_train = preprocess_image(train_folders, train_path)
X_test, y_test = preprocess_image(test_folders, test_path)

# Develop a sequential model using tensorflow keras
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size = 3, input_shape = (120, 160, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'tanh'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation = 'softmax')
])
    
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

result = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 25)
result = pd.DataFrame(result.history) 

def visualize(result, title, ylabel):
    for column in result:
        plt.plot(np.arange(25), result[column], label=column)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
visualize(result[['loss', 'val_loss']], 'Error of Training and Test Data', 'Error')
visualize(result[['accuracy', 'val_accuracy']], 'Accuracy of Training and Test Data', 'Accuracy')