# SIGNS-Hand-Gesture-Classification
A CNN-based approach to classifying hand signals using the SIGNS dataset
# SIGNS Hand Gesture Classification using CNN

This repository contains a Deep Learning implementation for classifying hand signals (representing numbers 0-5) using a Convolutional Neural Network (CNN). The project is built using TensorFlow and Keras within a Google Colab environment.

## Project Overview
The goal of this model is to recognize spatial patterns in image data. Unlike simple Dense networks, a CNN uses convolutional filters to identify edges, shapes, and textures, which is essential for accurate gesture recognition.

## Model Architecture
The model is built using a `Sequential` architecture with the following key components:
* **Conv2D Layers**: Two layers of convolution for feature extraction.
* **MaxPooling2D**: Downsampling to reduce computational load and highlight dominant features.
* **Flattening**: Converting the 2D feature maps into a 1D vector for classification.
* **Dropout (0.2)**: A regularization technique used to prevent overfitting by randomly deactivating neurons during training.
* **Softmax Output**: A 6-node dense layer providing probability scores for each class (0-5).

## Mathematical Foundation
The model optimizes weights $w$ and biases $b$ by minimizing the **Sparse Categorical Crossentropy** loss function. The update rule follows the Gradient Descent algorithm:

$$w := w - \alpha \frac{\partial J(w, b)}{\partial w}$$

Where:
* $\alpha$ is the learning rate.
* $\frac{\partial J}{\partial w}$ is the gradient of the cost function relative to the weights.

```## How to Run
1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run the `!wget` cell provided in the notebook to download the `.h5` datasets into the `/datasets` directory.
3. Execute the data loading and preprocessing cells.
4. Run the model training cell.

import os

# Create the directory structure
os.makedirs('datasets', exist_ok=True)

# Download the raw .h5 files directly to bypass LFS budget issues
!wget -O datasets/train_signs.h5 https://raw.githubusercontent.com/JudasDie/deeplearning.ai/master/Improving%20Deep%20Neural%20Networks/Week3/datasets/train_signs.h5
!wget -O datasets/test_signs.h5 https://raw.githubusercontent.com/JudasDie/deeplearning.ai/master/Improving%20Deep%20Neural%20Networks/Week3/datasets/test_signs.h5

```# Verify file sizes (~12MB for train)
!ls -lh datasets

## Results
After 30 epochs, the model achieves high training and validation accuracy, demonstrating effective generalization to unseen hand signals.
