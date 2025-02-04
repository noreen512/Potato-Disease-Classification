# Potato-Disease-Classification

Dataset: https://www.kaggle.com/arjuntejaswi/plant-village 
Colab Code: https://colab.research.google.com/drive/1DiRPNPBiK6EGIZBf3GbryIccK4L85bpo?usp=sharing

Project Description:
This project focuses on classifying potato leaf diseases using a deep learning model built with TensorFlow and Keras. The model is trained on the PlantVillage dataset, which contains images of healthy and diseased potato leaves. The goal is to develop an image classification system that can help farmers and agricultural experts detect potato diseases early.

🚀 Features
Utilizes TensorFlow and Keras for deep learning.
Processes images with Convolutional Neural Networks (CNNs).
Uses Matplotlib for visualization.
Handles dataset processing and augmentation.
Trains the model on a labeled dataset to classify different potato diseases.

It contains images categorized into:

Healthy Potato Leaves
Early Blight
Late Blight

🛠️ Technologies Used
Python
TensorFlow & Keras
Google Colab
Matplotlib
NumPy & Pandas

🏗️ Project Workflow
Dataset Loading & Preprocessing

Uploads and extracts the dataset.
Resizes images to a standard dimension (256x256 pixels).
Splits data into training and validation sets.


Model Architecture

Builds a CNN model with multiple convolutional and pooling layers.
Implements batch normalization and dropout for improved performance.


Training & Evaluation

Trains the model with an appropriate batch size and learning rate.
Evaluates the model's accuracy and loss.
Visualizes performance metrics.


Predictions

Tests the model on new images.
Outputs class predictions.


📈 Results & Insights
The model achieves high accuracy in distinguishing between healthy and diseased potato leaves.
Performance can be further improved with data augmentation and fine-tuning.
