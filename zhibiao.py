import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取两个模型的预测结果
data_model1 = pd.read_csv('../Transformer/data/transformer_predict.csv')  # 替换为实际路径
data_model2 = pd.read_csv('../LSSVM/data/lssvm_predictions.csv')  # 替换为实际路径

# 计算Kappa, OA, IoU指标的函数
def calculate_metrics(data):
    yu = 0.6  # 设定预测阈值

    # Kappa系数
    kappa = cohen_kappa_score(data['burned'], (data['predict'] >= yu).astype(int))

    # 总体准确率 (OA)
    oa = accuracy_score(data['burned'], (data['predict'] >= yu).astype(int))

    # 计算IoU（Intersection over Union）
    TP = ((data['predict'] >= yu) & (data['burned'] == 1)).sum()
    FP = ((data['predict'] >= yu) & (data['burned'] == 0)).sum()
    FN = ((data['predict'] < yu) & (data['burned'] == 1)).sum()
    TN = ((data['predict'] < yu) & (data['burned'] == 0)).sum()

    iou = TP / (TP + FP + FN)  # 计算IoU

    return kappa, oa, iou


# 计算两个模型的评估指标
metrics_model1 = calculate_metrics(data_model1)
metrics_model2 = calculate_metrics(data_model2)

# 设置指标名称
metrics = ['Kappa', 'OA', 'IoU']

# 设置模型1和模型2的指标值
model1_values = metrics_model1
model2_values = metrics_model2

# 打印两个模型的指标值
print(f"Transformer Model - Kappa: {metrics_model1[0]:.4f}, OA: {metrics_model1[1]:.4f}, IoU: {metrics_model1[2]:.4f}")
print(f"LSSVM Model------ - Kappa: {metrics_model2[0]:.4f}, OA: {metrics_model2[1]:.4f}, IoU: {metrics_model2[2]:.4f}")
