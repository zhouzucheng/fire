import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from numpy import vstack, argmax
from sklearn.metrics import accuracy_score
# Transformer模型定义
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes):
        super(TransformerClassifier, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        x = self.input_embedding(x)
        x = x.unsqueeze(1)  # 扩展维度，确保形状为 (sequence_length, batch_size, d_model)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Pooling
        x = self.fc(x)
        x = self.sigmoid(x)
        return x



