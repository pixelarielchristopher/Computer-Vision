import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Define data directories
defect_dir = r'C:\Users\pixel\PycharmProjects\pythonProject\Arsitektur IoT\Computer Vision\defect_photos'
undefect_dir = r'C:\Users\pixel\PycharmProjects\pythonProject\Arsitektur IoT\Computer Vision\undefect_photos'
processed_defect_dir = r'C:\Users\pixel\PycharmProjects\pythonProject\Arsitektur IoT\Computer Vision\processed_defect_photos'
processed_undefect_dir = r'C:\Users\pixel\PycharmProjects\pythonProject\Arsitektur IoT\Computer Vision\processed_undefect_photos'

# Define parameters
batch_size = 100
epochs = 50
img_height = 150
img_width = 150
test_size = 0.2

# Define preprocessing function to extract object from the image
def preprocess_image(image_path, output_dir):
    print("Image path:", image_path)
    img = cv2.imread(str(image_path))

    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 30, 255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert the mask so that white regions become foreground
    mask = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Convert the masked image to grayscale
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Combine the original image with the edges
    processed_img = cv2.bitwise_and(img, img, mask=edges)

    # Resize the image to a fixed size and convert to RGB
    resized_img = cv2.resize(processed_img, (img_height, img_width))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # Save the processed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, resized_img)
    
    return resized_img


# Load and preprocess images and labels
def load_images_and_labels(directory, output_dir):
    images = []
    labels = []
    label = 1 if directory == defect_dir else 0  # Set label based on directory
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            image_path = os.path.join(directory, file)
            try:
                preprocessed_img = preprocess_image(image_path, output_dir)
                images.append(preprocessed_img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
    return np.array(images), np.array(labels)

# Load and preprocess images and labels
defect_images, defect_labels = load_images_and_labels(defect_dir, processed_defect_dir)
not_defect_images, not_defect_labels = load_images_and_labels(undefect_dir, processed_undefect_dir)

# Concatenate defect and not defect images and labels
all_images = np.concatenate([defect_images, not_defect_images])
all_labels = np.concatenate([defect_labels, not_defect_labels])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels, test_size=test_size, random_state=42)

# Define preprocessing and augmentation for training data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Define preprocessing for validation data (only rescaling)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0)

# Define data generators for training and validation
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
validation_generator = validation_datagen.flow(X_test, y_test, batch_size=batch_size)

# Define the CNN model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model using data augmentation
history = model.fit(
    train_generator,
    steps_per_epoch=int(len(X_train) / batch_size),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(X_test) // batch_size)  # Use integer division here

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Save the model using native Keras format
model_path = r'C:\Users\pixel\PycharmProjects\pythonProject\Arsitektur IoT\Computer Vision\defect_classification_model.keras'
save_model(model, model_path)
print("Model saved as", model_path)
