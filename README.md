# Project on Feature Extraction on Indian Bank Currency Notes using Deep learning Technique.
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import layers
import cv2

# Define the directories for training and testing data
train_data_dir = r"D:\project\dataset\Train"
test_data_dir = r"D:\project\dataset\Test"

# Set up data generators for image loading and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load and preprocess data using the generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load the pre-trained VGG19 model
base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(224,224, 3))

# Freeze the layers of the pre-trained model
base_model.trainable = False
# Create a custom classification head
Classifier = tf.keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(units=4096, activation='relu'),
    layers.Dense(units=4096, activation='relu'),
    layers.Dense(units=7, activation='softmax')  # Change the number of units to match your number of classes
])

Classifier.compile(
    loss='sparse_categorical_crossentropy',  # Change to 'sparse_categorical_crossentropy'
    optimizer='adam',
    metrics=['accuracy']
)

# Fit the classifier
epochs = 50
r = Classifier.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
)

from tensorflow.keras.preprocessing import image
import numpy as np

# List of image paths for different currency notes
image_path = [
    r"D:\project\Dataset\Train\Real Currency\10-1 (14).jpg"
]
for img_path in image_path:
    # Load and preprocess the test image
    img = image.load_img(img_path, target_size=(224,224))
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis=0)

    # Use your classifier model to make predictions
    result = Classifier.predict(test_image)
    predicted_class_index = np.argmax(result, axis=1)
    predicted_class = class_names[predicted_class_index[0]]

    # Extract the true class from the image path (assuming the path structure is consistent)
    true_class = img_path.split('\\')[-2]

    # Calculate and print accuracy
    is_correct = (true_class == predicted_class)
    accuracy = 1 if is_correct else 0  # 0 if correct, 1 if incorrect

    if accuracy == 1:
        print("Real CURRENCY")
    else:
        print("Fake CURRENCY")

    print("------------------------")

