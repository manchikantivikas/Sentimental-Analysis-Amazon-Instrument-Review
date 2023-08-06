# Sentimental-Analysis-Amazon-Instrument-Review

# Sentiment Analysis on Amazon Musical Instrument Reviews

This repository contains the implementation of sentimental analysis on the Amazon Musical Instrument dataset. The analysis is conducted using various deep learning models and loss functions to understand and predict customer sentiments.

## Models Used
We employed four different models for the analysis. These include:
- **BERT (Bidirectional Encoder Representations from Transformers)**: Pre-trained transformer model for contextual embeddings.
- **LSTM (Long Short-Term Memory)**: A special kind of RNN that can capture long-term dependencies.
- **GRU (Gated Recurrent Unit)**: Simplified LSTM variant that often performs comparably.
- **CNN (Convolutional Neural Network)**: Used for sequence modeling, employing convolutional layers to capture local patterns.

## Loss Functions
The models were trained with three different loss functions to understand their impact on performance:
- **BCEWithLogitsLoss**: Combines a Sigmoid layer and the BCELoss in one single class. Useful for binary classification problems.
- **MSELoss**: Measures the mean squared error, suitable for regression problems.
- **L1Loss**: Measures the mean absolute error, robust to outliers.

## Optimizers
Two different optimizers were utilized to train the models:
- **Adam**: An adaptive learning rate optimizer known for its efficiency.
- **SGD (Stochastic Gradient Descent)**: A widely used method that updates the weights using only a subset of training data.

## Dataset
The dataset consists of Amazon Musical Instrument reviews, which includes textual reviews and corresponding sentiment labels.

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- scikit-learn
- Pandas
- NumPy

Project Report
For a comprehensive understanding of the methodologies used, experimental setups, results, insights, and detailed analysis, please refer to the project report attached in this repository.

Future Work
Experiment with different pre-processing techniques.
Investigate other loss functions and optimizers.
Explore ensemble methods.
