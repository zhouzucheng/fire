import os
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


csv_file = "../Data/wuding.csv"
data_sample = pd.read_csv(csv_file)
data = data_sample.sample(frac=0.5, random_state=42)

data = data[(data['dem'] != -9999) &
            (data['slope'] != -9999) &
            (data['aspect'] != -9999) &
            (data['landcover'] != -9999) &
            (data['ndvi'] != -9999) &
            (data['rhu'] != -9999) &
            (data['tem'] != -9999) &
            (data['burned'] != -9999)]

X = data[['dem', 'aspect', 'slope', 'landcover','ndvi','rhu','tem']].values
y = data['burned'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


input_dim = X_train.shape[1]
d_model = 64
nhead = 4
num_layers = 2
dropout = 0.1
learning_rate = 0.001
batch_size = 64
epochs = 50


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_projection(x)
        x += self.positional_encoding
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.fc(x)
        return self.sigmoid(x)



model = TransformerModel(input_dim, d_model, nhead, num_layers, dropout)


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_losses = []
train_accuracies = []


def train_model_with_plot(model, X_train, y_train, epochs, batch_size):
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%")


train_model_with_plot(model, X_train, y_train, epochs, batch_size)


def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).squeeze()
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
    return outputs

probabilities = test_model(model, X_test, y_test)

os.makedirs("data", exist_ok=True)
torch.save(model.state_dict(), "data/transformer_model.pth")
print("Model saved to 'data/transformer_model.pth'.")

os.makedirs("images", exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")

# 绘制损失函数图
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.savefig(f"images/training_loss/{timestamp}.png")
print(f"Loss plot saved to 'images/training_loss/{timestamp}.png'.")
