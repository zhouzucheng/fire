import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import os
from datetime import datetime

csv_file = '../Transformer/data/transformer_predict.csv'
df = pd.read_csv(csv_file)


y = df.iloc[:, 0]
X = df.iloc[:, 1:]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.fc2 = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.unsqueeze(0)
        x = self.transformer(x, x)
        x = self.fc2(x.squeeze(0))
        return torch.sigmoid(x)


input_dim = X_train.shape[1]
d_model = 64
nhead = 4
num_layers = 2
dropout = 0.1


model = TransformerModel(input_dim, d_model, nhead, num_layers, dropout)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 50
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()

    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(y_train.values, dtype=torch.float32)

    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs.squeeze(), targets)

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = (outputs.squeeze() > 0.5).float()
        accuracy = (preds == targets).float().mean()

    train_losses.append(loss.item())
    train_accuracies.append(accuracy.item())

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


plt.savefig(f'../Transformer/images/training_loss/Training_Loss_{timestamp}.png', format='png', bbox_inches='tight')

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.savefig(f'../Transformer/images/training_loss/Training_Accuracy_{timestamp}.png', format='png', bbox_inches='tight')

plt.show()

def model_wrapper(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x_tensor).squeeze().numpy().reshape(-1, 1)


background = X_train


explainer = shap.Explainer(model_wrapper, background)


shap_values = explainer(X_train)


plt.figure()
shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="dot", show=False)


plt.savefig(f'../Transformer/images/shap/SHAP_Summary_Plot_{timestamp}.png', format='png', bbox_inches='tight')

plt.figure(figsize=(10, 5), dpi=1200)
shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="bar", show=False)
plt.title('../Transformer/SHAP Sorted Feature Importance')
plt.tight_layout()

plt.savefig(f'../Transformer/images/shap/SHAP_Feature_Importance_{timestamp}.png', format='png', bbox_inches='tight')
plt.show()