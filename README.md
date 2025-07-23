# TensorFlow MNIST Demo

This repository contains implementations of neural networks for the MNIST handwritten digit recognition task using TensorFlow.

## Implementations

### Simple Dense Network (`mnist_simple.py`)

A basic implementation using fully connected (dense) layers.

```bash
python mnist_simple.py
```

### CNN Implementation (`mnist_cnn.py`)

An improved implementation using Convolutional Neural Networks, which are more suitable for image data. This implementation includes:

- Convolutional layers with batch normalization
- Dropout for regularization
- Data augmentation option
- Early stopping and learning rate scheduling
- TensorBoard integration
- Model checkpointing

#### Usage

```bash
# Basic usage
python mnist_cnn.py

# With data augmentation
python mnist_cnn.py --use_data_augmentation

# Custom hyperparameters
python mnist_cnn.py --batch_size 64 --learning_rate 0.0005 --epochs 15

# View all options
python mnist_cnn.py --help
```

#### Visualizing Training with TensorBoard

After training, you can visualize the training metrics using TensorBoard:

```bash
tensorboard --logdir=logs/fit
```

Then open your browser to http://localhost:6006 to view the TensorBoard dashboard.

## Performance Comparison

The CNN implementation typically achieves 99%+ accuracy on the MNIST test set, compared to around 98% for the simple dense network, due to the convolutional architecture being more suitable for image data.

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization, optional)
