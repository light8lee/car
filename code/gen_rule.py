import rule_utils as ru
import pandas as pd
import numpy as np

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

matchers = [(sub, ru.match_subject('../rules/{}.txt'.format(sub))) for sub in subjects]
train_df = pd.read_csv('../data/input2.csv')
test_df = pd.read_csv('../data/test_public.csv')

rule_train = pd.DataFrame()
for sub in subjects:
    rule_train[sub] = np.array([0] * train_df.shape[0])

for i, value in enumerate(train_df['content']):
    for sub, matcher in matchers:          
        if matcher(value):
            rule_train.loc[i, sub] = 1

rule_train.to_csv('../data/rule_train.csv', index=False) 

rule_test = pd.DataFrame()
for sub in subjects:
    rule_test[sub] = np.array([0] * train_df.shape[0])

for i, value in enumerate(test_df['content']):
    for sub, matcher in matchers:          
        if matcher(value):
            rule_test.loc[i, sub] = 1

rule_test.to_csv('../data/rule_test.csv', index=False) 