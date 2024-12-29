# # flask_predict.py
# import numpy as np
# from flask import Flask, request, jsonify
# import torch
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import csv
# from model import TransformerClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
#
# app = Flask(__name__)
#
# class FireDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = torch.tensor(features, dtype=torch.float32)
#         self.labels = torch.tensor(labels.values, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]
#
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.data.decode('utf-8')
#
#     data['landcover'] = data['landcover'].astype(float)
#     data['dem'] = data['dem'].astype(float)
#     data['slope'] = data['slope'].astype(float)
#     data['aspect'] = data['aspect'].astype(float)
#     labels = data['burned']
#
#     X = data[['landcover', 'dem', 'slope', 'aspect']]
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#
#     predict_dataset = FireDataset(X, labels)
#     predict_loader = DataLoader(predict_dataset, batch_size=1024, shuffle=False)
#
#     all_labels = []
#     all_preds = []
#
#     model = torch.load('fire.pth')
#     with torch.no_grad():
#         for inputs, labels in predict_loader:
#             outputs = model(inputs)
#             preds = outputs.squeeze()
#             all_labels.extend(labels.numpy())
#             all_preds.extend(preds.numpy())
#     threshold = 0.5
#     binary_preds = [1 if pred > threshold else 0 for pred in all_preds]
#
#     # 计算准确率
#     accuracy = accuracy_score(all_labels, binary_preds)
#     for pred in all_preds:
#         print(pred)
#     data['predict'] = all_preds
#     data.to_csv('newFire.csv')
#     print(f"Accuracy on the predict set: {accuracy:.4f}")
#
#
# if __name__ == "__main__":
#     # import sys
#     # model_path = sys.argv[1]
#     # model = torch.jit.load(model_path)
#     model = torch.load('fire.pth')
#     model.eval()
#     app.run(port=5000)


from flask import Flask, request, jsonify
import torch
from model import TransformerClassifier  # Assuming TransformerClassifier is your model class
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your trained PyTorch model
model_path = 'fire.pth'
model = torch.load(model_path)
model.eval()


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request JSON data
    data = request.get_json()

    # Extract parameters from the JSON data
    landcover = data['landcover']
    dem = data['dem']
    slope = data['slope']
    aspect = data['aspect']

    # Perform any necessary data preprocessing
    # Convert input data to the format your model expects
    # Example: Convert inputs to PyTorch tensor
    inputs = torch.tensor([[landcover, dem, slope, aspect]], dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        output = model(inputs)
        prediction=output.item()

    # Return the prediction as JSON response
    return jsonify({'burn_probability': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
