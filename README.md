# Waste Sorting Using Computer Vision with MobileNet

This project demonstrates how we can use computer vision and machine learning to automate the process of sorting waste items into two categories: Organic and Recyclable. Using a pre-trained MobileNet model, we train a classification model that can accurately predict whether an image represents an organic waste item or a recyclable one.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Objectives](#objectives)
3. [Dataset Information](#dataset-information)
4. [Understanding MobileNet](#understanding-mobilenet)
5. [Code Implementation](#code-implementation)
6. [Testing the Model](#testing-the-model)
7. [Conclusion](#conclusion)

---

## 1. Problem Statement

Imagine if every piece of trash in the world could automatically find the right bin to go to! That’s the magic of computer vision with waste sorting. Sorting waste is essential for keeping our planet clean. When we sort waste correctly:
- **Recyclables** (like plastic bottles and cans) can be reused, which saves resources.
- **Organic waste** can be composted, helping plants grow.
- **Non-recyclables** go to landfills, but it’s better if we minimize them.

By training a computer to recognize waste, we can speed up sorting and reduce mistakes, helping the environment and the recycling process.

---

## 2. Objectives

In this project, we will:
- Use computer vision to help identify types of waste.
- Train a model called MobileNet to "see" and categorize images as **Organic (O)** or **Recyclable (R)**.
- Use a dataset of waste images, practice sorting them, and observe how the computer learns to differentiate between them.

---

## 3. Dataset Information

Our dataset contains images of waste items that belong to two categories:
- **O (Organic)**: For food scraps, leaves, or other natural waste.
- **R (Recyclable)**: For plastics, metals, and items that can be reused.

This dataset is organized into two folders:
- **Train Folder**: Where images are used to teach the computer what each type of waste looks like.
- **Test Folder**: To check if the computer has learned to sort images correctly.

### Dataset Download

You can download the dataset manually from Kaggle:
- **[Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)**

---

## 4. Understanding MobileNet

MobileNet is a lightweight and efficient model that performs well for image classification tasks, especially in environments with limited computational resources, such as mobile devices. 

### Key Features:
- **Lightweight**: Designed for fast performance on mobile devices.
- **Good Accuracy**: Efficiently classifies objects like recyclable or organic waste.
- **Feature Extraction**: Extracts key details from images (e.g., shapes and colors).

### MobileNet Variants:
- **MobileNetV1**: Introduced depthwise separable convolutions.
- **MobileNetV2**: Improved with inverted residuals and linear bottlenecks.
- **MobileNetV3**: Optimized further with neural architecture search (NAS).

---

## 5. Code Implementation

### Import Libraries and Prepare Dataset

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up paths to the train and test directories
base_dir = "/content/waste-classification-data/DATASET/"
train_dir = base_dir + "TRAIN"
test_dir = base_dir + "TEST"

# Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_data = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
```

### MobileNet Model Configuration

```python
mobilenet_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
mobilenet_model.trainable = False

# Add custom layers for classification
model = tf.keras.Sequential([
    mobilenet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Train the Model

```python
history = model.fit(train_data, validation_data=test_data, epochs=10)

# Save the trained model
model.save('mobilenet_waste_classifier.h5')
```

---

## 6. Testing the Model

You can test the model with a sample image by using the following code:

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_path = 'mobilenet_waste_classifier.h5'
test_image_path = r"Dataset/Test/R/R_10399.jpg"

# Load the model and prepare the image
model = tf.keras.models.load_model(model_path)
test_img = load_img(test_image_path, target_size=(224, 224))
test_img_array = img_to_array(test_img) / 255.0
test_img_array = np.expand_dims(test_img_array, axis=0)

# Make a prediction
predicted_prob = model.predict(test_img_array)[0][0]
predicted_class = 'Organic' if predicted_prob < 0.5 else 'Recyclable'

# Display the result
plt.imshow(test_img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()
```

---

## 7. Conclusion

This project demonstrates the use of computer vision and MobileNet to classify waste items into categories like Organic and Recyclable. By automating waste sorting, this model can help improve recycling efforts and contribute to a cleaner environment. The approach can eventually be used in various applications such as waste management systems and recycling plants.

---
This structure provides a clear and concise overview of your project, steps to implement, and how to test the model.
