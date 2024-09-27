# Adarsh Kumar | akumar39@u.rochester.edu

import numpy as np

class MultilayerPerceptron:
    def __init__(self, learning_rate=0.01, epochs=1000, hidden_units=10, regularization=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.regularization = regularization

    def fit(self, X, T, X_val, T_val):
        self.num_classes = T.shape[1]
        self.num_features = X.shape[1]
        self.weights1 = np.random.randn(self.num_features, self.hidden_units) * 0.01
        self.biases1 = np.zeros((1, self.hidden_units))
        self.weights2 = np.random.randn(self.hidden_units, self.num_classes) * 0.01
        self.biases2 = np.zeros((1, self.num_classes))
        self.train_accuracies = []
        self.val_accuracies = []

        for epoch in range(self.epochs):
            hidden_layer = np.tanh(np.dot(X, self.weights1) + self.biases1)
            logits = np.dot(hidden_layer, self.weights2) + self.biases2
            probs = self.softmax(logits)
            loss = -np.mean(np.sum(T * np.log(probs), axis=1)) + self.regularization * (np.sum(self.weights1 ** 2) + np.sum(self.weights2 ** 2))
            grad_logits = probs - T
            grad_w2 = np.dot(hidden_layer.T, grad_logits) / X.shape[0] + 2 * self.regularization * self.weights2
            grad_b2 = np.sum(grad_logits, axis=0, keepdims=True) / X.shape[0]
            grad_hidden = np.dot(grad_logits, self.weights2.T) * (1 - hidden_layer ** 2)
            grad_w1 = np.dot(X.T, grad_hidden) / X.shape[0] + 2 * self.regularization * self.weights1
            grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True) / X.shape[0]
            self.weights1 -= self.learning_rate * grad_w1
            self.biases1 -= self.learning_rate * grad_b1
            self.weights2 -= self.learning_rate * grad_w2
            self.biases2 -= self.learning_rate * grad_b2

            train_acc = self.accuracy(self.predict(X), T)
            val_acc = self.accuracy(self.predict(X_val), T_val)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

    def predict(self, X):
        hidden_layer = np.tanh(np.dot(X, self.weights1) + self.biases1)
        logits = np.dot(hidden_layer, self.weights2) + self.biases2
        probs = self.softmax(logits)
        return probs

    def save(self, path):
        np.savez(path, weights1=self.weights1, biases1=self.biases1, weights2=self.weights2, biases2=self.biases2)

    @staticmethod
    def load(path):
        data = np.load(path)
        model = MultilayerPerceptron()
        model.weights1 = data['weights1']
        model.biases1 = data['biases1']
        model.weights2 = data['weights2']
        model.biases2 = data['biases2']
        return model

    def softmax(self, a):
        ea = np.exp(a - np.max(a, axis=1, keepdims=True))
        return ea / np.sum(ea, axis=1, keepdims=True)

    def accuracy(self, y_pred, y_true):
        return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)) / y_true.shape[0]