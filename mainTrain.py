import cv2
import os
import tensorflow as tf
import keras
from tensorflow import keras 
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.utils import to_categorical 

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import accuracy_score

#from tensorflow.keras.utils import normalize

image_directory = 'dataset/'

no_tumor_images = os.listdir(image_directory + 'NO/')
yes_tumor_images = os.listdir(image_directory + 'YES/')

dataset = []

label = []

#INPUT_SIZE = 64
INPUT_SIZE = 128
#print(no_tumor_images)

#path = 'IMG-0147-00012 (2).jpg'

#print(path.split('.')[1])

for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg' :
        image = cv2.imread(image_directory + 'NO/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg' :
        image = cv2.imread(image_directory + 'YES/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

#print(len(dataset))
#print(len(label))

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

#Reshape = (n, image_width, image_height, n_channel)
#print(x_train.shape)
#x_train = 
#import keras
#x_train = normalize(x_train, axis = 1)
#x_test = normalize(x_test, axis = 1)
#y_train = to_categorical(y_train,num_classes=2)
#y_test = to_categorical(y_test,num_classes=2)
x_train = np.array(x_train, dtype=float)
x_test = np.array(x_test, dtype=float)

x_train_normalized = (x_train / 255).astype('float32')
x_test_normalized = (x_test / 255).astype('float32')

y_train = to_categorical(y_train,num_classes=2)
y_test = to_categorical(y_test,num_classes=2)



#Model Building

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Flatten())
#model.add(Dense(64)) # nen de 256 để tối ưu 
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))


optimizers1 = keras.optimizers.Adam(learning_rate = 1e-3)

model.compile(loss='categorical_crossentropy', 
              #optimizer='adam', 
              optimizer = optimizers1, 
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, min_lr = 1e-6)

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

import log

file_logging_callback = log.FileLoggingCallback()

#them learning rate vào optimizer,  decade rate
#regularization hàm loss bớt phụ thuộc 
model.fit(x_train_normalized,
          y_train, 
          batch_size=16,
          verbose = 1, 
          epochs = 15, #early stoping 
          validation_data=(x_test_normalized, y_test),
          #shuffle=False,
          shuffle = True,
          callbacks = [file_logging_callback, reduce_lr, early_stopping])


model.save('BrainTumor10EpochsCategorical_2.h5')
# Thay vì lưu dưới dạng .h5, sử dụng định dạng .keras
#model.save('my_model.keras')
