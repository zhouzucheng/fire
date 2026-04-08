import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df1 = pd.read_csv('../Transformer/data/transformer_predict.csv')
df3 = pd.read_csv('../LSSVM/data/lssvm_predictions.csv')

y_pred1 = df1['predict']
y_true1 = df1['burned']

y_pred3 = df3['predict']
y_true3 = df3['burned']

fpr1, tpr1, _ = roc_curve(y_true1, y_pred1)
roc_auc1 = auc(fpr1, tpr1)
fpr3, tpr3, _ = roc_curve(y_true3, y_pred3)
roc_auc3 = auc(fpr3, tpr3)

plt.figure()
plt.plot(fpr1, tpr1, color='blue', lw=2, label='Transformer (AUC = %0.2f)' % roc_auc1)
plt.plot(fpr3, tpr3, color='red', lw=2, label='LSSVM (AUC= %0.2f)' % roc_auc3)
# plt.plot(fpr2, tpr2, color='green', lw=2, label='ChuXiong (AUC = %0.2f)' % roc_auc2)


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')

plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
