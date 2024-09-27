# Adarsh Kumar | akumar39@u.rochester.edu

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization

    def fit(self, X, T, X_val, T_val):
        self.num_classes = T.shape[1]
        self.num_features = X.shape[1]
        self.weights = np.random.randn(self.num_features, self.num_classes) * 0.01
        self.biases = np.zeros((1, self.num_classes))
        self.train_accuracies = []
        self.val_accuracies = []

        for epoch in range(self.epochs):
            logits = np.dot(X, self.weights) + self.biases
            probs = self.softmax(logits)
            loss = -np.mean(np.sum(T * np.log(probs), axis=1)) + self.regularization * np.sum(self.weights ** 2)
            grad_w = np.dot(X.T, (probs - T)) / X.shape[0] + 2 * self.regularization * self.weights
            grad_b = np.sum(probs - T, axis=0, keepdims=True) / X.shape[0]
            self.weights -= self.learning_rate * grad_w
            self.biases -= self.learning_rate * grad_b

            train_acc = self.accuracy(self.predict(X), T)
            val_acc = self.accuracy(self.predict(X_val), T_val)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.biases
        probs = self.softmax(logits)
        return probs

    def save(self, path):
        np.savez(path, weights=self.weights, biases=self.biases)

    @staticmethod
    def load(path):
        data = np.load(path)
        model = LogisticRegression()
        model.weights = data['weights']
        model.biases = data['biases']
        return model

    def softmax(self, a):
        ea = np.exp(a - np.max(a, axis=1, keepdims=True))
        return ea / np.sum(ea, axis=1, keepdims=True)

    def accuracy(self, y_pred, y_true):
        return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)) / y_true.shape[0]