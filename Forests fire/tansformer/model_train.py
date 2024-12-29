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
# filePath = '../xiangyun/data/dongchuan.csv'
filePath = '../data/newFire.csv'
fire = []

data=pd.read_csv(filePath)
data['landcover']=data['landcover'].astype(float)
data['dem']=data['dem'].astype(float)
data['slope']=data['slope'].astype(float)
data['aspect']=data['aspect'].astype(float)
labels=data['burned']
X=data[['landcover','dem','slope','aspect']]
# train_data, test_data, train_labels, test_labels = train_test_split(fire, labels, test_size=0.2,
#                                                                  random_state=42)

#过滤掉空值
# filtered_data = data[~((data['landcover'] == -9999) | (data['dem'] == -9999) | (data['slope'] == -9999) | (data['aspect'] == -9999) | (data['burned'] == -9999))]
def calculate_accuracy(outputs, targets):
    """
    计算模型预测的准确率。

    参数:
    - outputs: 模型预测的输出张量，可以是概率值或类别预测。
    - targets: 真实标签的张量。

    返回:
    - accuracy: 计算出的准确率（百分比）。
    """
    # 如果输出是概率值，则转换为类别预测
    if outputs.dim() > 1:
        outputs = torch.round(outputs)
    else:
        outputs = (outputs >= 0.5).float()

    correct = (outputs == targets).float().sum()
    accuracy = correct / targets.size(0) * 100.0
    return accuracy.item()


# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 创建自定义数据集类
class FireDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = FireDataset(X_train, y_train)
test_dataset = FireDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 模型参数
input_dim = X_train.shape[1]
d_model = 64
nhead = 64
num_layers = 2
dim_feedforward = 64
num_classes = 1

# # 模型参数
# input_dim = X_train.shape[1]
# d_model = 64
# nhead = 4
# num_layers = 2
# dim_feedforward = 64
# num_classes = 1

# 创建模型
model = TransformerClassifier(input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes)
# model=Transformer(d_model,num_layers,nhead,64,64,2048)
# 训练模型
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

trainloss = []
testloss = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    test_loss=0.0

    trainAcc=0
    testAcc=0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        # outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        trainAcc+=calculate_accuracy(outputs,labels)
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = outputs.squeeze()
        loss = criterion(preds, labels)
        test_loss += loss.item() * inputs.size(0)
        testAcc+=calculate_accuracy(outputs, labels)
    epoch_loss = running_loss / len(train_loader.dataset)
    testLoss=test_loss/len(test_loader.dataset)
    trainloss.append(epoch_loss)
    testloss.append(testLoss)
    print(f'Epoch {epoch + 1}/{num_epochs}, TrainLoss: {epoch_loss:.4f},TestLoss:{testLoss:.4f}')
    # print("Train accuracy: %.5f" % (trainAcc / len(train_loader.dataset)))
    # print("Test accuracy: %.5f" % (testAcc/len(test_loader.dataset)))


# 可视化
plt.figure(figsize=(10, 7))
plt.plot(trainloss, label='loss')
plt.xlabel('Steps')
plt.ylabel('trainLoss')
plt.savefig("../image/trainloss.png")

plt.figure(figsize=(10, 7))
plt.plot(testloss, label='loss')
plt.xlabel('Steps')
plt.ylabel('testLoss')
plt.savefig("../image/testloss.png")

plt.show()
# 评估模型
model.eval()
all_labels = []
all_testpreds = []
all_trainpreds=[]
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = outputs.squeeze()
        all_labels.extend(labels.numpy())
        all_testpreds.extend(preds.numpy())
    for inputs, labels in train_loader:
        outputs = model(inputs)
        preds = outputs.squeeze()
        all_trainpreds.extend(preds.numpy())

#
# torch.save(model,'../xiangyun/data/fire.pth')
# 将预测的概率转换为二进制标签
threshold = 0.5
binary_preds = [1 if pred > threshold else 0 for pred in all_testpreds]

# 计算准确率
accuracy = accuracy_score(all_labels, binary_preds)


print(f"Accuracy on the test set: {accuracy:.4f}")

# 打印测试集上的前几组数据的预测概率
print("Predicted probabilities for the first few test samples:", all_testpreds[:5])

# data['predict']=all_testpreds
# data.to_csv('newFire.csv',index=False)
