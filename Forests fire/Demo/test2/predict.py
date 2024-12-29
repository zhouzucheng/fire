import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from model import TransformerClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载和预处理数据
filePath = '../data/dongchuan.csv'
fire = []

data=pd.read_csv(filePath)
data['landcover']=data['landcover'].astype(float)
data['dem']=data['dem'].astype(float)
data['slope']=data['slope'].astype(float)
data['aspect']=data['aspect'].astype(float)
labels=data['burned']
X=data[['landcover','dem','slope','aspect']]

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 创建自定义数据集类
class FireDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return self.features[idx], self.labels[idx]


predict_dataset=FireDataset(X,labels)
predict_loader=DataLoader(predict_dataset,batch_size=1024,shuffle=False)

all_labels=[]
all_preds=[]

#model=torch.load('fire.pth')
model=torch.load('fire.pth')
with torch.no_grad():
    for inputs, labels in predict_loader:
        outputs = model(inputs)
        preds = outputs.squeeze()
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())
threshold = 0.5
binary_preds = [1 if pred > threshold else 0 for pred in all_preds]

# 计算准确率
accuracy = accuracy_score(all_labels, binary_preds)
for pred in all_preds:
    print(pred)
data['predict']=all_preds
data.to_csv('../data/transformer.csv',index=False)
print(f"Accuracy on the predict set: {accuracy:.4f}")