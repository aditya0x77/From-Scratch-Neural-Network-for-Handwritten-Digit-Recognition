import numpy as np
import pandas as pd

def sigmoid(z):
    z = np.clip(z, -500, 500)
    result = 1/(1+np.exp(-z))
    return result

def init_params():
    W1 = np.random.randn(784, 128) * np.sqrt(1/784)
    b1 = np.zeros(128)
    W2 = np.random.randn(128, 64) * np.sqrt(1/128)
    b2 = np.zeros(64)
    W3 = np.random.randn(64, 10) * np.sqrt(1/64)
    b3 = np.zeros(10)
    return W1, b1, W2, b2, W3, b3

def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out

def dense_vectorized(a_in, W, B):
    Z = np.dot(a_in, W) + B
    a_out = sigmoid(Z)
    return a_out


def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((y.size, num_classes)) 
    one_hot_y[np.arange(y.size), y] = 1         
    return one_hot_y

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return e / np.sum(e, axis=1, keepdims=True)


def compute_loss(Softmaxed_final_output, Y_encoded):
    m = Y_encoded.shape[0]
    eps = 1e-9
    loss = -np.sum(Y_encoded * np.log(Softmaxed_final_output + eps)) / m
    return loss

def forward_backward_pass(X_batch, Y_encoded_batch, weights, biases, LR=0.01):

    A1 = dense_vectorized(X_batch, weights[0], biases[0])
    A2 = dense_vectorized(A1, weights[1], biases[1])
    Z3 = np.dot(A2, weights[2]) + biases[2]
    A3 = softmax(Z3)

    m = X_batch.shape[0]
    delta3 = (A3 - Y_encoded_batch) / m    
    dW3 = np.dot(A2.T, delta3)  
    db3 = np.sum(delta3, axis=0)

    dA2 = np.dot(delta3, weights[2].T)
    delta2 = dA2 * (A2 * (1 - A2)) 
    dW2 = np.dot(A1.T, delta2)       
    db2 = np.sum(delta2, axis=0)

    dA1 = np.dot(delta2, weights[1].T)
    delta1 = dA1 * (A1 * (1 - A1))
    dW1 = np.dot(X_batch.T, delta1) 
    db1 = np.sum(delta1, axis=0)

    weights[0] -= LR * dW1
    biases[0] -= LR * db1
    weights[1] -= LR * dW2
    biases[1] -= LR * db2
    weights[2] -= LR * dW3
    biases[2] -= LR * db3

    loss = compute_loss(A3, Y_encoded_batch)
    return weights, biases, loss

def predict(X, weights, biases):
    Z1 = np.dot(X, weights[0]) + biases[0]
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, weights[1]) + biases[1]
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, weights[2]) + biases[2]
    A3 = softmax(Z3)

    return np.argmax(A3, axis=1)

def predict_single(x, y_true, weights, biases):

    x = x.reshape(1, -1)
    pred = predict(x, weights, biases)[0]

    print(f"True Label     : {y_true}")
    print(f"Predicted Label: {pred}")

    return pred