# Importing libraries
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image

# Image data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
dataset_path = 'archive'

# Training set (use the entire dataset for training)
training_set = train_datagen.flow_from_directory(
    directory=dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=True  # Shuffle the data for better training
)

# Building the CNN model
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the model
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
cnn.fit(x=training_set, epochs=25)

# Save the trained model
cnn.save('pothole_detection_model.keras')
