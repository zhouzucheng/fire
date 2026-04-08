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

# 1. 数据预处理
csv_file = '../Transformer/data/transformer_predict.csv'  # 请替换为实际路径
df = pd.read_csv(csv_file,nrows=200)

# 处理缺失值（如果有的话），去除无效行
df = df[df[['dem', 'aspect', 'slope', 'landcover', 'ndvi', 'rhu', 'tem', 'burned']].notnull().all(axis=1)]

# 将数据拆分为特征和标签
X = df[['dem', 'slope', 'aspect', 'landcover',  'ndvi', 'rhu', 'tem']]  # 特征
y = df['burned']  # 标签

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. 构建Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.fc2 = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.fc1(x)  # 输入层
        x = x.unsqueeze(0)  # 增加一个维度，适应Transformer的输入
        x = self.transformer(x, x)  # 通过Transformer
        x = self.fc2(x.squeeze(0))  # 输出层
        return torch.sigmoid(x)  # 输出概率


# 参数设置
input_dim = X_train.shape[1]  # 输入特征维度
d_model = 64  # Transformer模型的维度
nhead = 4  # 注意力头数
num_layers = 2  # Transformer层数
dropout = 0.1  # dropout比例

# 实例化模型
model = TransformerModel(input_dim, d_model, nhead, num_layers, dropout)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练模型
epochs = 50
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()

    # 转换为PyTorch的Tensor
    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(y_train.values, dtype=torch.float32)

    # 清空梯度
    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs.squeeze(), targets)

    # 反向传播
    loss.backward()
    optimizer.step()

    # 计算准确率
    with torch.no_grad():
        preds = (outputs.squeeze() > 0.5).float()
        accuracy = (preds == targets).float().mean()

    # 存储每个epoch的损失和准确率
    train_losses.append(loss.item())
    train_accuracies.append(accuracy.item())

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

# 4. 绘制训练图和损失函数图
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 获取当前时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# 保存图像到根目录的 data 文件夹，并加上时间戳
plt.savefig(f'../Transformer/images/training_loss/Training_Loss_{timestamp}.png', format='png', bbox_inches='tight')

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

# 保存准确率图像
plt.savefig(f'../Transformer/images/training_loss/Training_Accuracy_{timestamp}.png', format='png', bbox_inches='tight')

plt.show()

# 创建模型包装器以处理numpy输入
def model_wrapper(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x_tensor).squeeze().numpy().reshape(-1, 1)  # 保证输出为二维数组

# 定义背景数据集
background = X_train  # 确保这行代码存在且未被注释

# 使用SHAP的标准方法初始化解释器
explainer = shap.Explainer(model_wrapper, background)

# 计算SHAP值
shap_values = explainer(X_train)

# 绘制特征重要性图
plt.figure()
shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="dot", show=False)

# 保存SHAP图像
plt.savefig(f'../Transformer/images/shap/SHAP_Summary_Plot_{timestamp}.png', format='png', bbox_inches='tight')

plt.figure(figsize=(10, 5), dpi=1200)
shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="bar", show=False)
plt.title('../Transformer/SHAP Sorted Feature Importance')
plt.tight_layout()

# 保存条形图SHAP图像
plt.savefig(f'../Transformer/images/shap/SHAP_Feature_Importance_{timestamp}.png', format='png', bbox_inches='tight')

plt.show()
