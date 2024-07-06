# SimpleCNN Model for Image Classification

## Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification on the Fashion MNIST dataset. The project showcases the architecture of the SimpleCNN model, training procedures, hyperparameters, and the results of different model versions, demonstrating various deep learning techniques and improvements.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Hyperparameters and Training Philosophy](#hyperparameters-and-training-philosophy)
- [Results Comparisons](#results-comparisons)
- [Conclusions](#conclusions)
- [Further Improvements](#further-improvements)
- [Contact](#contact)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/simplecnn-image-classification.git
    cd simplecnn-image-classification
    ```

2. **Install required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the training script, use the following command:

```bash
python train.py
```
Make sure to configure any necessary paths or parameters within the train.py script.

## Model Architecture

The architecture of the SimpleCNN model is designed to capture hierarchical features of images through a series of convolutional, activation, and pooling layers, followed by fully connected layers for classification.

### Initial Model
- Convolutional Layers: Three convolutional layers (conv1, conv2, conv3) with ReLU activations and max pooling.
- Fully Connected Layers: Transition to fully connected layers (fc1, fc2) with a final softmax activation for class probabilities.

```bash
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x
```

## Hyperparameters and Training Philosophy
- Batch Size: 64
- Learning Rate: 0.001
- Epochs: 10

The choice of these hyperparameters balances computational efficiency and the need for effective learning.

## Results Comparisons

### Initial Architecture
- First Run:
  - Epoch 1: Test Accuracy: 75.74%
  - Epoch 10: Test Accuracy: 88.99%
  
- Final Run:
  - Epoch 1: Test Accuracy: 90.72%
  - Epoch 10: Test Accuracy: 90.35%

### Improved Architecture
- First Run:
  - Epoch 1: Test Accuracy: 86.39%
  - Epoch 10: Test Accuracy: 90.82%
  
- Final Run:
  - Epoch 1: Test Accuracy: 92.43%
  - Epoch 10: Test Accuracy: 90.72%

## Conclusions
- Learning Speed: The improved architecture converges faster.
- Robustness: The improved architecture consistently outperforms the initial architecture.
- Model Persistence: The improved architecture maintains higher accuracy across runs.

## Further Improvements
- Architectural Complexity: Explore adjustments like filter sizes and residual connections.
- Hyperparameter Tuning: Fine-tune learning rate, batch size, and optimizer settings.
- Ensemble Methods: Combine predictions from multiple models.
- Regularization Techniques: Introduce dropout or weight decay to prevent overfitting.

## Contact

If you have any questions or suggestions, feel free to reach out:

- Jonathan Nazareth
- Email: jgnazareth26@gmail.com
- LinkedIn: https://www.linkedin.com/in/jonathan-nazareth/
- GitHub: https://github.com/Jonathan2603
