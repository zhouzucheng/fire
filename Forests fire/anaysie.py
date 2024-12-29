import csv

import jieba
import jieba.analyse
import pandas as pd
from  DataPreProcess import standardProcess
goodreviews=[]
badreviews=[]
with open('predict.csv', encoding='ANSI', errors='ignore', newline='') as f:
    csvObject = csv.reader(f)
    for row in csvObject:
        if row[0]=='正面':
            goodreviews.append(row[1])
        else:
            badreviews.append(row[1])
# goodreviews=standardProcess.standardPosseg(goodreviews)
# badreviews=standardProcess.standard(badreviews)
# print(goodreviews)
# print(badreviews)
goodtext=''
badrtext=''
for goodreview in goodreviews:
    goodtext+=goodreview
for badreview in badreviews:
    badrtext+=badreview
emotionTag={'PA':'快乐','PE':'安心','PD':'尊敬','PH':'赞扬','PG':'相信','PB':'喜爱','PK':'祝愿',
                'NA':'愤怒','NB':'悲伤','NJ':'失望',
                'NH':'疚','PF':'思','NI':'慌','NC':'恐惧','NG':'羞','NE':'烦闷','ND':'憎恶','NN':'贬责',
                'NK':'妒忌','NL':'怀疑', 'PC':'惊奇'}
emotionalVocabularys=[]
comments=[]
filePath='G:\PythonProject\\NlpProject\DataPreProcess\SentimentAnalysisDictionary-main\情感词汇本体/情感词汇本体.csv'
df = pd.read_csv(filePath,encoding='ANSI')
words=[]
emotional=[]
strength=[]
polarity=[]
goodtag=[]
badtag=[]
for index, row in df.iterrows():
    words.append(row['词语'])
    emotional.append(row['情感分类'])
    strength.append(row['强度'])
    polarity.append(row['极性'])
    emotionalVocabularys.append([row['词语'],row['情感分类'],row['强度'],row['极性']])

for goodreview in goodreviews:
    comment=jieba.cut(goodreview)
    for word in comment:
        if words.count(word)!=0:
            goodtag.append(emotional[words.index(word)])
            break
for badreview in badreviews:
    comment=jieba.cut(badreview)
    for word in comment:
        if words.count(word)!=0:
            badtag.append(emotional[words.index(word)])
            break
goodlast=[]
badlast=[]

for tag in goodtag:
    for key in emotionTag:
        if tag==key:
            goodlast.append(emotionTag[key])
            break
for tag in badtag:
    for key in emotionTag:
        if tag==key:
            badlast.append(emotionTag[key])
            break
print('正面评论情感为：',goodlast,'占比为:',len(goodlast)/(len(goodlast)+len(badlast)))
for tag in emotionTag:
    print('正面评论中{}的占比为：'.format(emotionTag[tag]),goodlast.count(emotionTag[tag])/len((goodlast))*100,'%')
print('负面评论情感为：',badlast,'占比为:',len(badlast)/(len(goodlast)+len(badlast)))
for tag in emotionTag:
    print('负面评论中{}的占比为：'.format(emotionTag[tag]),badlast.count(emotionTag[tag])/len((badlast))*100,'%')
# TF/IDF算法
keywords = jieba.analyse.extract_tags(sentence=goodtext,
                                          topK=100,
                                          withWeight=True)
print('TF/IDF提取文本结果：{}'.format(keywords))
keywords2=jieba.analyse.extract_tags(sentence=badrtext,
                                          topK=100,
                                          withWeight=True)
print('TF/IDF提取文本结果：{}'.format(keywords2))
# keywords= jieba.analyse.textrank(comments, topK=10, withWeight=True)
# print('textrank提取文本结果：{}'.format(keywords))