import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import pickle
import os
import pandas as pd
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
device = get_device()
print("Device: ", device)


class ExtendedMNISTDataset(Dataset):
    def __init__(self, root: str = "/kaggle/input/fii-nn-2025-homework-4", train: bool = True):
        file = "extended_mnist_test.pkl"
        if train:
            file = "extended_mnist_train.pkl"
        file = os.path.join(root, file)
        with open(file, "rb") as fp:
            self.data = pickle.load(fp)

    def __len__(self, ) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]


class TensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]


class MyModel(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden1_size)
        self.bn_1 = nn.BatchNorm1d(hidden1_size)
        self.drop_1 = nn.Dropout(p=0.1)

        self.layer_2 = nn.Linear(hidden1_size, hidden2_size)
        self.bn_2 = nn.BatchNorm1d(hidden2_size)
        self.drop_2 = nn.Dropout(p=0.1)

        self.layer_3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x: Tensor):
        x = self.layer_1(x)
        x = x.relu()
        x = self.bn_1(x)
        x = self.drop_1(x)

        x = self.layer_2(x)
        x = x.relu()
        x = self.bn_2(x)
        x = self.drop_2(x)

        x = self.layer_3(x)
        return x


model = MyModel(input_size=784, hidden1_size=1024, hidden2_size=512, output_size=10).to(device)
print("My model: ", model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)


def processBatch(data, device):
    return data.to(device, dtype=torch.float32)


def trainEpoch(model, train_dataLoader, criterion, optimizer, device):
    model.train()
    mean_loss = 0.0

    for data, labels in train_dataLoader:
        data = processBatch(data, device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        mean_loss = mean_loss + loss.item()
    mean_loss = mean_loss / len(train_dataLoader)
    return mean_loss


@torch.inference_mode()
def validate(model, val_dataLoader, criterion, device):
    model.eval()
    mean_loss = 0.0
    correct = 0
    total = 0

    for data, labels in val_dataLoader:
        data = processBatch(data, device)
        labels = labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)

        mean_loss = mean_loss + loss.item()
        _, preds = outputs.max(1)
        correct = correct + (preds == labels).sum().item()
        total = total + labels.size(0)
    mean_loss = mean_loss / len(val_dataLoader)
    acc = correct / total
    return mean_loss, acc

def main(model, train_dataLoader,val_dataLoader, criterion, optimizer,scheduler, device, epochs):
    with tqdm(range(epochs)) as tbar:
        for epoch in tbar:
            train_loss = trainEpoch(model, train_dataLoader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_dataLoader, criterion, device)

            scheduler.step()

            tbar.set_description(
                f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.3f} | "
                f"Val: {val_loss:.3f} | Acc: {val_acc:.4f}"
            )


train_data = []
train_labels = []
for image, label in ExtendedMNISTDataset(train=True):
    train_data.append(image)
    train_labels.append(label)

test_data = []
for image, label in ExtendedMNISTDataset(train=False):
    test_data.append(image)

train_data = np.array(train_data, dtype = np.float32)
train_labels = np.array(train_labels, dtype = np.int64)
test_data = np.array(test_data, dtype = np.float32)

train_data = train_data.reshape(train_data.shape[0], -1) #(N,28,28) -> (N, 784)
test_data = test_data.reshape(test_data.shape[0], -1)

train_data = train_data / 255.0
test_data = test_data / 255.0

mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0) + 1e-6

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

train_data = torch.from_numpy(train_data)
train_labels = torch.from_numpy(train_labels)
test_data = torch.from_numpy(test_data)

full_train_dataset = TensorDataset(train_data, train_labels)
n_total = len(full_train_dataset)
n_validate = int(n_total * 0.1)
n_train = n_total - n_validate

train_dataset, validate_dataset = random_split(
    full_train_dataset,
    [n_train, n_validate],
    generator = torch.Generator().manual_seed(42)
)

batch_size = 128

train_dataLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
validate_dataLoader = DataLoader(validate_dataset, batch_size = 2*batch_size, shuffle = False)

dummy_label = torch.zeros(len(test_data), dtype = torch.long)
test_dataset = TensorDataset(test_data, dummy_label)
test_dataLoader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


epochs = 25
main(model, train_dataLoader, validate_dataLoader,  criterion, optimizer,scheduler, device, epochs)

model.eval()
predictions = []

with torch.inference_mode():
    for data, _ in test_dataLoader:
        data = data.to(device)
        outputs = model(data)
        _, preds = outputs.max(1)
        predictions.extend(preds.cpu().numpy())