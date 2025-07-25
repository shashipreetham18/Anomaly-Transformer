
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from AnomalyTransformer import AnomalyTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("datasets/ServerMachineDataset/train/machine-1-1.txt")
X = df.values
X = np.pad(X, ((0, 0), (0, 56 - X.shape[1])), mode='constant')

X_train=X
X_train = torch.FloatTensor(X_train)

batch_size = 1024
train_data = TensorDataset(X_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

N = X_train.shape[0]
d_model = X_train.shape[1]

model = AnomalyTransformer(N, d_model, hidden_dim=64)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, optimizer, epochs):
    print(f"traing started! ")
    model.train()
    losses = []
    for epoch in range(epochs):
        print(f"epoch: {epoch} started")
        epoch_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            model.to(device)
            data = torch.FloatTensor(data[0]).to(device)
            x_hat, P_list, S_list = model(data)
            min_loss = model.min_loss(x_hat, data, P_list, S_list)
            max_loss = model.max_loss(x_hat, data, P_list, S_list)

            if torch.isnan(min_loss) or torch.isnan(max_loss):
                print("Encountered NaN in loss. Skipping this batch.")
                continue

            min_loss.backward(retain_graph=True)
            max_loss.backward()
            optimizer.step()
            epoch_loss += min_loss.item()

        losses.append(epoch_loss/len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'anomaly_transformer_weights.pth')
    return losses

losses=train(model, train_loader, optimizer, epochs=50)