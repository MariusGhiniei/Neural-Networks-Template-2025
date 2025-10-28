import pickle
import os
import pandas as pd
import numpy as np

# train_file = "/kaggle/input/fii-nn-2025-homework-2/extended_mnist_train.pkl"
# test_file = "/kaggle/input/fii-nn-2025-homework-2/extended_mnist_test.pkl"

train_file = "Data/extended_mnist_train.pkl"
test_file = "Data/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = []
for image, label in test:
    test_data.append(image.flatten())

import numpy as np

X_train = np.stack(train_data).astype(np.float32) / 255.0   #(m, 784)
y_train = np.array(train_labels, dtype=np.int64)            #(m,)
X_test  = np.stack(test_data).astype(np.float32) / 255.0    #(m_test, 784)

mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

num_classes = 10
num_features = X_train.shape[1]

#Targeting the label
Y_train = np.eye(num_classes, dtype=np.float32)[y_train]    #(m, 10)

rng = np.random.default_rng(12)

#init the W(784, 10) and b(10,)
W = 0.01 * rng.standard_normal((num_features, num_classes), dtype=np.float32)
b = np.zeros((num_classes,), dtype=np.float32)

def softmax(x:np.ndarray) -> np.ndarray:
    #return e^(x_i)/sum_{j=1}^K e^x_j
    z = x - np.max(x, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def accuracy(X : np.ndarray, Y : np.ndarray, W : np.ndarray, b:np.ndarray) -> float:
    #return accuracy in range [0,1]
    probs = softmax(X @ W + b)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(Y, axis=1)
    return (y_pred == y_true).mean()

#Hyper - parameters
epochs = 30
batch_size = 128
lr = 0.2
l2 = 3e-5

m = X_train.shape[0]
size_epoch = (m + batch_size - 1) // batch_size

for epoch in range(1, epochs + 1):
    #randomize the data
    index = rng.permutation(m)
    X_rand = X_train[index]
    Y_rand = Y_train[index]

    epoch_loss = 0.0

    for i in range(size_epoch):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, m)
        Xb = X_rand[start_index : end_index]
        Yb = Y_rand[start_index : end_index]
        B = Xb.shape[0]

        #Forward
        logits = Xb @ W + b
        probs = softmax(logits)

        #log-stability
        eps = 1e-8
        cross_entropy = -np.sum(Yb * np.log(probs + eps)) / B
        #l2 regularization for W
        loss = cross_entropy + 0.5 * l2 * np.sum(W * W)
        epoch_loss = epoch_loss + loss

        #Backward
        der_logits = (probs - Yb) / B
        der_W = Xb.T @ der_logits + l2 * W
        der_b = np.sum(der_logits, axis=0)

        # Update
        W = W - lr * der_W
        b = b - lr * der_b
        lr = 0.1 * (.95 ** epoch)

    train_acc = accuracy(X_train, Y_train, W, b)
    print(f"Epoch {epoch:02d}/{epochs}  loss={epoch_loss/size_epoch:.4f}  acc={train_acc:.4f}")

# Predict on test set
test_probs = softmax(X_test @ W + b)
predictions = np.argmax(test_probs, axis=1)

# This is how you prepare a submission for the competition
predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)