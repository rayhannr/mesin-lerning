# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (320, 240, 3), activation = 'relu'))
#bikin 32 3x3 feature detector
#images will be converted to 3d array if colored, 2d if grayscale
#input_shape maksudnya 3d array (colored image) with 64x64 for the dimension
#using relu to prevent pixels with negative value in order to have non linearity karena image processing itu non linear

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#ambil max value dari sub matrix 2x2

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
#unitnya gak terlalu kecil biar modelnya bagus dan gak terlalu gede biar computingnya gak berat. common practicenya pake pangkat 2
classifier.add(Dense(units = 3, activation = 'sigmoid'))
#pake sigmoid karena outcomenya cuma binary. kalo gak binary pake softmax

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#loss pake binary karena outcomenya binary, kalo lebih dari 2 pake categorical

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, #scale pixel value to between 0 and 1
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('flowers/training_set',
                                                 target_size = (320, 240),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('flowers/test_set',
                                            target_size = (320, 240),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 240, #number of images in the training set
                         epochs = 4,
                         validation_data = test_set,
                         validation_steps = 60) #number of images in the test set