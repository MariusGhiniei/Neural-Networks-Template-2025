import pickle
import pandas as pd
import numpy as np

train_file = "Data/extended_mnist_train.pkl"
test_file  = "Data/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)
with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = [image.flatten() for image, _ in test]

train_data = np.array(train_data, dtype=np.float32) / 255.0
train_labels = np.array(train_labels, dtype=np.int64)
test_data    = np.array(test_data,  dtype=np.float32) / 255.0

X = train_data.astype(np.float32)
y = train_labels.astype(np.int64)
X_test = test_data.astype(np.float32)

mean = X.mean(axis=0, keepdims=True)
std  = X.std(axis=0, keepdims=True) + 1e-10
X = (X - mean) / std
X_test = (X_test - mean) / std

#for validation
rng = np.random.default_rng(1234)
idx = np.arange(len(X))
rng.shuffle(idx)
split = int(0.95 * len(idx))
Xtr, Ytr = X[idx[:split]], y[idx[:split]]
Xval, Yval = X[idx[split:]], y[idx[split:]]

input_dim = X.shape[1]
hidden_dim = 100
num_classes = int(y.max()) + 1


W1 = rng.normal(0, np.sqrt(1.0 / input_dim), size=(input_dim, hidden_dim)).astype(np.float32)
b1 = np.zeros((1, hidden_dim), dtype=np.float32)

W2 = rng.normal(0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, num_classes)).astype(np.float32)
b2 = np.zeros((1, num_classes), dtype=np.float32)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def reluGrad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def softmax(x:np.ndarray) -> np.ndarray:
    #return e^(x_i)/sum_{j=1}^K e^x_j
    z = x - np.max(x, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def getClass(y, C):
    matZero = np.zeros((len(y), C), dtype=np.float32)
    matZero[np.arange(len(y)), y] = 1.0
    return matZero

#hyper-parameters
epochs = 170
batch  = 1024
lr = 0.18
momentum = 0.9
l2 = 5e-4

#buffer momentum
vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)


for epoch in range(1, epochs+1):
    order = rng.permutation(len(Xtr))
    Xtr, Ytr = Xtr[order], Ytr[order]

    for s in range(0, len(Xtr), batch):
        xBatch = Xtr[s:s+batch]
        yBatch = Ytr[s:s+batch]
        m = xBatch.shape[0]
        yBatchClass = getClass(yBatch, num_classes)

        #forward
        z1 = xBatch @ W1 + b1
        a1 = relu(z1)
        scores = a1 @ W2 + b2
        probs = softmax(scores)

        #backward
        dScores = (probs - yBatchClass) / m
        dW2 = a1.T @ dScores + l2 * W2
        db2 = np.sum(dScores, axis=0, keepdims=True)

        da1 = dScores @ W2.T
        dz1 = da1 * reluGrad(z1)
        dW1 = xBatch.T @ dz1 + l2 * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # momentum update
        vW2 = momentum * vW2 - lr * dW2
        W2 = W2 + vW2
        vb2 = momentum * vb2 - lr * db2
        b2 = b2 + vb2
        vW1 = momentum * vW1 - lr * dW1
        W1 = W1 + vW1
        vb1 = momentum * vb1 - lr * db1
        b1 = b1 + vb1

    def eval(Xs, Ys):
        z1 = Xs @ W1 + b1
        a1 = relu(z1)
        logits = a1 @ W2 + b2
        probs = softmax(logits)
        yClass= getClass(Ys, num_classes)
        eps = 1e-8
        ce = -np.mean(np.sum(yClass * np.log(probs + eps), axis=1))
        l2reg = 0.5 * l2 * (np.sum(W1*W1) + np.sum(W2*W2))
        loss = ce + l2reg
        acc = (np.argmax(probs, axis=1) == Ys).mean()
        return loss, acc

    trainLoss, trainAcc = eval(Xtr, Ytr)
    valLoss, valAcc = eval(Xval, Yval)

    print(f"Epoch {epoch:02d} | train_loss={trainLoss:.4f} acc={trainAcc*100:.2f}% | "
          f"val_loss={valLoss:.4f} acc={valAcc*100:.2f}% | lr={lr:.4f}")

# test the MLP
z1 = X_test @ W1 + b1
a1 = relu(z1)
probs = softmax(a1 @ W2 + b2)
predictions = np.argmax(probs, axis=1)

predictions_csv = {
    "ID": [],
    "target": [],
}
for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission2.csv", index=False)
