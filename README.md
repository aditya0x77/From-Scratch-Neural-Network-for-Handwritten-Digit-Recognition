# Neural Network from Scratch for Handwritten Digit Recognition

A fully connected neural network built entirely from scratch using **NumPy** and **Pandas**. This project recreates the core components of a modern neural network—including forward propagation, backpropagation, gradient descent, and weight updates—without using deep learning frameworks such as TensorFlow or PyTorch.

The model is trained on the MNIST handwritten digit dataset and achieves approximately **92% classification accuracy**, demonstrating how neural networks can be implemented using only fundamental numerical operations.

---

## Highlights

- Built a feedforward neural network entirely from scratch
- Implemented forward propagation and backpropagation without automatic differentiation
- Implemented mini-batch gradient descent for training
- Used softmax activation with cross-entropy loss for multi-class classification
- Achieved **~92% test accuracy** on the MNIST dataset

---

## Network Architecture

| Layer | Configuration |
|------|---------------|
| Input | 784 features (28 × 28 grayscale image) |
| Hidden Layer 1 | 128 neurons (Sigmoid) |
| Hidden Layer 2 | 64 neurons (Sigmoid) |
| Output | 10 neurons (Softmax) |

---

## Sample Inputs

The network is trained to recognize handwritten digits from the MNIST dataset.

<p align="center">
  <img src="images/5.png" width="140">
  <img src="images/0.png" width="140">
  <img src="images/4.png" width="140">
</p>

---

## Learning Objectives

This project was built to understand the internal mechanics of neural networks by implementing every stage of training manually. Rather than relying on high-level libraries, all computations—including gradient calculation, parameter updates, and optimization—are performed using only NumPy.

---

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

---

## End of the Line, Choomba

Thanks for checking out the project.

See you in the next one.

<p align="center">
  <img src="images/Johnny.gif" width="450">
</p>
