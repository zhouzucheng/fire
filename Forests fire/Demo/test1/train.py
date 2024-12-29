import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

# 读取CSV文件
data = pd.read_csv('../data/dongchuan.csv')

# 定义特征和标签
X = data[['dem', 'slope', 'aspect', 'landcover']]
y = data['burned']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义LSSVM模型（使用RBF核函数）并指定参数（sigma = 1 对应 gamma = 0.5）
best_model = SVC(kernel='rbf', C=10, gamma=2, probability=True)
best_model.fit(X_train, y_train)

# 手动设置决策函数的b值
best_model.intercept_ = [-0.9512]


# 预测并评估模型
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 计算并打印准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 计算AUC值
y_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc}")

# 保存模型到本地
joblib_file = "fire.pth"
joblib.dump(best_model, joblib_file)
print(f"Model saved to {joblib_file}")
