# exLattice: A Pure C Neural Network from Scratch

A minimal, educational implementation of a feedforward neural network in C. This project was built to gain a deep, first-principles understanding of backpropagation and gradient descent by implementing the underlying mathematics without the abstraction of modern libraries.

## 🚀 Overview

This is my first attempt at implementing the backpropagation algorithm from scratch. While production-grade libraries like PyTorch or TensorFlow are significantly faster (utilizing GPUs and highly optimized kernels), **exLattice** is designed for educational transparency. Every weight update, activation calculation, and error propagation is handled manually in pure C.

### Current Features:
- **Architecture**: Multi-layer feedforward neural network.
- **Activations**: ReLU (hidden layers) and identity (output).
- **Optimization**: Stochastic Gradient Descent (SGD) with batch support.
- **Initialization**: He Initialization (specifically tuned for ReLU stability).
- **Loss Function**: Mean Squared Error (MSE).
- **Dataset**: Built-in support for loading and training on the **MNIST digits dataset**.

## 🧠 Why C?

Writing a neural network in C forces you to confront challenges that are often hidden by high-level frameworks.
And frankly because come on, I get to say I made a neural network in C.

### 2. Setup the Data
Download the MNIST training files into a `data/` directory:

### 3. Build and Run
```bash
make
bin/app
```

## 📈 Learning Progress

The model currently achieves a steady decrease in loss over 10 epochs on the MNIST dataset:
- **Epoch 0 Loss**: ~0.62
- **Epoch 9 Loss**: ~0.18

## 📝 Roadmap & Future Improvements
- [ ] Implement Cross-Entropy Loss for better classification performance.
- [ ] Add Softmax activation to the output layer.
- [ ] Add Gradient Clipping to improve stability with higher learning rates.
- [ ] Support for saving/loading trained models to `.models` files.
- [ ] Implement an evaluation mode to calculate accuracy % on a test set.

## AI Notice
- Antigravity and Gemini were used to assist with fixing memory leaks, writing this README and loading the dataset.
- But the logic and mathematics implementation of the Backpropagation algorithm and the Neural Network structure were completly deduced and implemented by me.

---
*Disclaimer: This project is for educational purposes only. If you need speed, use a GPU-accelerated framework!*
