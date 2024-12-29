import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# 加载之前训练好的模型
joblib_file = "fire.pth"
loaded_model = joblib.load(joblib_file)

# 读取新的CSV文件
new_data = pd.read_csv('../data/dongchuan.csv')

# 定义特征（确保特征名称与训练时一致）
X_new = new_data[['dem', 'slope', 'aspect', 'landcover']]

# 数据标准化（使用与训练时相同的标准化方法）
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# 使用加载的模型进行预测
y_new_proba = loaded_model.predict_proba(X_new_scaled)[:, 1]

# 将预测结果添加到新数据中
new_data['predict'] = y_new_proba




# 将新数据保存到新的CSV文件
new_data.to_csv('../data/lssvm.csv', index=False)

print("Predictions saved to ../data/predicted_new_data.csv")
