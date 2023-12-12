# Performance Comparison of Convolutional Neural Network Training on CPU and GPU

## Overview

This project aims to evaluate the computational efficiency of training Convolutional Neural Networks (CNNs) by comparing two distinct hardware configurations: an Intel Core i5 13th Gen CPU with 12 cores and an Nvidia RTX 3050 GPU. The focus is on image classification tasks using the CIFAR-10 dataset.

## Features

- **Hardware Configurations:**
  - CPU: Intel Core i5 13th Gen with 12 cores
  - GPU: Nvidia RTX 3050

- **Deep Learning Framework:**
  - TensorFlow and Keras

- **Parallelization Strategy:**
  - Utilizes `tf.distribute.MirroredStrategy()` for GPU parallelization

- **Optimization Frameworks:**
  - CUDA and cuDNN for GPU optimization

## Dataset

The project utilizes the CIFAR-10 dataset, a well-established benchmark for image classification. CIFAR-10 consists of 60,000 32x32 color images across 10 classes, making it ideal for evaluating the generalization capabilities of neural network models.

## Key Metrics

- **Training Time:** Measured in seconds for 2 epochs
- **Model Accuracy:** Percentage of correct predictions
- **Efficiency:** A measure of performance efficiency, calculated based on training time and hardware configuration

## Results

### CPU Computation
- **Training Time (2 epochs):** 652 seconds
- **Model Accuracy:** 67%
- **Efficiency:** 52

### GPU Computation
- **Training Time (2 epochs):** 115 seconds
- **Model Accuracy:** 66.8%
- **Efficiency:** 115

## Conclusion

The GPU implementation using the Nvidia RTX 3050 demonstrated significant speedup and efficiency compared to the CPU counterpart. The training time was substantially reduced, resulting in an efficient utilization of the GPU parallelization capabilities. The achieved accuracy on the GPU is comparable to the CPU, indicating that the parallelized training did not compromise model performance.

## Getting Started

### Prerequisites

- Python
- TensorFlow and Keras
- CUDA and cuDNN for GPU optimization

### Installation

1. Clone the repository: `git clone github.com/amith-2001/CPU_vs_GPU.git`
2. Install dependencies: `pip install -r requirements.txt`

### Usage

1. Run the main scripts: CPU_compute.py and GPU_compute.py
2. View the results in the generated reports.

## Future Improvements

- Experiment with different CNN architectures
- Explore training for additional epochs
- Conduct further hardware comparisons
