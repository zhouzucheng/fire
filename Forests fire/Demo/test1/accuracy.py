import pandas as pd

# 读取 CSV 文件
data = pd.read_csv('../data/lssvm.csv')

# 对 predict 列进行分类
data['predict_class'] = data['predict'].apply(lambda x: 0 if 0 <= x < 0.5 else 1)

# 计算准确率
correct_predictions = (data['predict_class'] == data['burned']).sum()
total_predictions = len(data)
accuracy = correct_predictions / total_predictions

print(f"准确率为：{accuracy}")