import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('out_name')
args = parser.parse_args()

subjects = {
    'price': '价格', 
    'interior': '内饰', 
    'power': '动力', 
    'surface': '外观', 
    'safety': '安全性', 
    'operation':'操控', 
    'gas' : '油耗', 
    'space': '空间', 
    'comfort': '舒适性', 
    'config': '配置'
}

result = pd.DataFrame()
test = pd.read_csv('../data/test_public.csv')
result['content_id'] = test['content_id']
for name, sub in subjects:
    xgb_df = pd.read_csv('../tmp/xgb-{}.csv'.format(name))
    lgb_df = pd.read_csv('../tmp/lgb-{}.csv'.format(name))
    fuse = (xgb_df['pos_proba'] + lgb_df['pos_proba'] - (xgb_df['neg_proba'] + lgb_df['neg_proba'])) > 0
    result[sub] = fuse

with open('../output/{}v1.csv'.format(args.out_name), 'w', encoding='utf-8') as f:
        line = '{},{},0,'
        f.write('content_id,subject,sentiment_value,sentiment_word')
        for index, row in result.iterrows():
            has = False
            for sub in subjects.values():
                if row[sub]:
                    has = True
                    value = line.format(row['content_id'], sub)
                    f.write('\n')
                    f.write(value)
            if not has: # 没有主题，默认值为价格主题，情感为０
                value = line.format(row['content_id'], '价格')
                f.write('\n')
                f.write(value)
