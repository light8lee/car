import pandas as pd
import numpy as np
import jieba
from collections import defaultdict

def get_deafult():
    return defaultdict(int)
table2 = defaultdict(get_deafult)
table4 = defaultdict(get_deafult)


# 用于生成4分类和2分类的训练标签数据
with open('../data/train.csv', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        content_id, content, subject, sentiment_value, sentiment_word = line.split(',')
        table2['content'][content_id] = content
        table2[subject][content_id] = 1

        table4['content'][content_id] = content
        table4[subject][content_id] = int(sentiment_value)+2 # -1， 0， 1 -> 1，2，3

input2 = pd.DataFrame.from_dict(table2)
input2.fillna(0, inplace=True)
input2.rename_axis('content_id', inplace=True)

input2.to_csv('../data/input2.csv')

input4 = pd.DataFrame.from_dict(table4)
input4.fillna(0, inplace=True)
input4.rename_axis('content_id', inplace=True)

input4.to_csv('../data/input4.csv')

for col in ['价格', '内饰', '动力', '外观', '安全性', '操控', '油耗', '空间', '舒适性', '配置']:
    pos = input2[input2[col]!=0].shape[0]
    neg = input2.shape[0] - pos
    print("'{}': {},".format(col, neg/pos))