import csv

import torch
import torch as T
import numpy as np
import torch.nn.functional as F

from model import TextClassifier
from text_featuring import load_file_file, text_feature

model = T.load('pth/dianping.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label_dict, char_dict = load_file_file()
label_dict_rev = {v: k for k, v in label_dict.items()}
print(label_dict)
print(char_dict)
filePath='../DataSpider2/3A-LaoYvHe/data/laoyvhe_dianping.csv'
reviews=[]
with open(filePath, encoding='ANSI', errors='ignore', newline='') as f:
    csvObject = csv.reader(f)
    for row in csvObject:
        reviews.append(row[2])
label=[]
for review in reviews:

    labels, contents = ['正面'], [review]
    samples, y_true = text_feature(labels, contents, label_dict, char_dict)
    print(samples)
    print(len(samples[0]))
    x = T.from_numpy(np.array(samples)).long()
    y_pred = model(x).to(device)
    print(y_pred)
    y_numpy = F.softmax(y_pred, dim=1).detach().cpu().numpy()
    print(y_numpy)
    predict_list = np.argmax(y_numpy, axis=1).tolist()
    for i, predict in enumerate(predict_list):
        print(f"第{i+1}个文本，预测标签为： {label_dict_rev[predict]}")
        label.append(label_dict_rev[predict])
fileObject = open('./DataInference/LaoYvHe_predict.csv', 'a', encoding='ANSI', errors='ignore', newline='')
reviewsInfo = csv.writer(fileObject)
# reviewsInfo.writerow(["label", "reviews"])
flag=0

for review in reviews:
        reviewsInfo.writerow([label[flag], review])
        flag+=1