# Adarsh Kumar | akumar39@u.rochester.edu

from util import *
from logreg import LogisticRegression
from mlp import MultilayerPerceptron
import matplotlib.pyplot as plt

# To load and normalize datasets from the data folder
X_loans_train, T_loans_train = loadDataset('../data/loans.train')
X_loans_val, T_loans_val = loadDataset('../data/loans.val')
X_water_train, T_water_train = loadDataset('../data/water.train')
X_water_val, T_water_val = loadDataset('../data/water.val')

X_loans_train = normalize(X_loans_train)
X_loans_val = normalize(X_loans_val)
X_water_train = normalize(X_water_train)
X_water_val = normalize(X_water_val)

T_loans_train = toHotEncoding(T_loans_train, 2)
T_loans_val = toHotEncoding(T_loans_val, 2)
T_water_train = toHotEncoding(T_water_train, 2)
T_water_val = toHotEncoding(T_water_val, 2)

# Excute and print Logistic Regression accuracy on Loans Dataset
logreg_loans = LogisticRegression(learning_rate=0.01, epochs=1000, regularization=0.01)
logreg_loans.fit(X_loans_train, T_loans_train, X_loans_val, T_loans_val)
acc_loans_train_logreg = logreg_loans.train_accuracies[-1]
acc_loans_val_logreg = logreg_loans.val_accuracies[-1]
print(f'Logistic Regression Accuracy on Loans Training Set: {acc_loans_train_logreg}')
print(f'Logistic Regression Accuracy on Loans Validation Set: {acc_loans_val_logreg}')

# Excute and print Logistic Regression accuracy on Water Dataset
logreg_water = LogisticRegression(learning_rate=0.01, epochs=1000, regularization=0.01)
logreg_water.fit(X_water_train, T_water_train, X_water_val, T_water_val)
acc_water_train_logreg = logreg_water.train_accuracies[-1]
acc_water_val_logreg = logreg_water.val_accuracies[-1]
print(f'Logistic Regression Accuracy on Water Training Set: {acc_water_train_logreg}')
print(f'Logistic Regression Accuracy on Water Validation Set: {acc_water_val_logreg}')

# Excute and print Multilayer Perceptron accuracy on Loans Dataset
mlp_loans = MultilayerPerceptron(learning_rate=0.01, epochs=1000, hidden_units=20, regularization=0.01)
mlp_loans.fit(X_loans_train, T_loans_train, X_loans_val, T_loans_val)
acc_loans_train_mlp = mlp_loans.train_accuracies[-1]
acc_loans_val_mlp = mlp_loans.val_accuracies[-1]
print(f'Multilayer Perceptron Accuracy on Loans Training Set: {acc_loans_train_mlp}')
print(f'Multilayer Perceptron Accuracy on Loans Validation Set: {acc_loans_val_mlp}')
mlp_loans.save("../src/loans.model.npz")

# Excute and print Multilayer Perceptron accuracy on Water Dataset
mlp_water = MultilayerPerceptron(learning_rate=0.01, epochs=1000, hidden_units=20, regularization=0.01)
mlp_water.fit(X_water_train, T_water_train, X_water_val, T_water_val)
acc_water_train_mlp = mlp_water.train_accuracies[-1]
acc_water_val_mlp = mlp_water.val_accuracies[-1]
print(f'Multilayer Perceptron Accuracy on Water Training Set: {acc_water_train_mlp}')
print(f'Multilayer Perceptron Accuracy on Water Validation Set: {acc_water_val_mlp}')
mlp_water.save("../src/water.model.npz")

# To plot accuracy vs. compute graph for all models and datasets combinations
plt.figure(figsize=(12, 8))

# Logistic Regression - Loans
plt.plot(logreg_loans.train_accuracies, label='LogReg Loans - Train')
plt.plot(logreg_loans.val_accuracies, label='LogReg Loans - Val')

# Logistic Regression - Water
plt.plot(logreg_water.train_accuracies, label='LogReg Water - Train')
plt.plot(logreg_water.val_accuracies, label='LogReg Water - Val')

# Multilayer Perceptron - Loans
plt.plot(mlp_loans.train_accuracies, label='MLP Loans - Train')
plt.plot(mlp_loans.val_accuracies, label='MLP Loans - Val')

# Multilayer Perceptron - Water
plt.plot(mlp_water.train_accuracies, label='MLP Water - Train')
plt.plot(mlp_water.val_accuracies, label='MLP Water - Val')

# Customize plot
plt.xlabel('Compute')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Compute for Different Models and Datasets')
plt.legend()
plt.grid(True)
plt.show()
