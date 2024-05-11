import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
width, height = 100, 100  # Define your desired fixed size

input_shape = (width, height, 1)  # Channels should be 1 since images are grayscale

# Define data generators for training and validation
train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.3, color_mode='edges')

batch_size = 32

train_generator = train_data_gen.flow_from_directory(
    directory=r'C:/Users/pixel/PycharmProjects/pythonProject/Arsitektur IoT/Computer Vision',
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='training'  # Specify subset as 'training' for the training data
)

validation_generator = train_data_gen.flow_from_directory(
    directory=r'C:/Users/pixel/PycharmProjects/pythonProject/Arsitektur IoT/Computer Vision',
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    subset='validation'  # Specify subset as 'validation' for the validation data
)

# Define the CNN model
def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification: defect or no defect
    return model

# Create the model
model = create_model(input_shape)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          epochs=10,
          validation_data=validation_generator)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print("Validation Accuracy:", accuracy)
