import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np
import os

# Define new variable names and paths
data_directory = r"C:\Users\alodie\OneDrive\Desktop\aifinal"
train_data_dir = os.path.join(data_directory, "training")
verify_data_dir = os.path.join(data_directory, "verification")
image_path = os.path.join(data_directory, "test", "testBenign.jpg")

num_classes = 2
target_size = (224, 224)  # Change this to your desired target size

# Define preprocessing and augmentation options
data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

batch_size = 32

# Load and preprocess training data
train_generator = data_generator.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load and preprocess validation data
validation_generator = data_generator.flow_from_directory(
    verify_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Evaluate the model
evaluation = model.evaluate(validation_generator)
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

# Save and load the model
model.save("image_classification_model.h5")
loaded_model = load_model("image_classification_model.h5")

# Load and preprocess a new image
def preprocess_new_image(image_path):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Preprocess new image and make a prediction
new_image = preprocess_new_image(image_path)
predictions = loaded_model.predict(new_image)

# Define class names
class_names = ['malignant', 'benign']

# Get the predicted class index and name
predicted_class_index = np.argmax(predictions)
predicted_class_name = class_names[predicted_class_index]

# Print the predictions
print("Predicted class probabilities:", predictions)
print("Predicted class:", predicted_class_name)
