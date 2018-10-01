
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import xgboost as xgb
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('subject')
# args = parser.parse_args()


# In[ ]:


X = pd.read_csv('../data/merge_train.csv')
X.head(1)


# In[ ]:


Y_all = pd.read_csv('../data/input.csv')
Y_all.head(1)


# ### 验证

# In[ ]:


# In[ ]:


seed = 7
# test_size = 1000
# X_train, X_test, y_train, y_test = train_test_split(X, Y_all, test_size=test_size, random_state=seed)


# In[ ]:


params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 4,               # 类别数，与 multisoftmax 并用
    'tree_method': 'gpu_hist',
#     'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
#     'max_depth': 12,               # 构建树的深度，越大越容易过拟合
#     'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#     'subsample': 0.7,              # 随机采样训练样本
#     'colsample_bytree': 0.7,       # 生成树时进行的列采样
#     'min_child_weight': 3,
#     'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
#     'eta': 0.007,                  # 如同学习率
#     'seed': 1000,
#     'nthread': 4,                  # cpu 线程数
}


# In[ ]:

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

def f1_eval(trues, preds):
    label = preds.get_label()
    score = f1_score(trues, labels, average='macro')
    return 'f1_eval', score

print('validating...')
for sub in subjects.values():
    dtrain = xgb.DMatrix(X, Y_all[sub])
    num_rounds = 500
    result = xgb.cv(params, dtrain, num_rounds, nfold=10, maximize=True, feval=f1_eval)
    print(result)


'''
test_df = pd.read_csv('../data/merge_test.csv')
result = pd.DataFrame()
test = pd.read_csv('../data/test_public.csv')
result['content_id'] = test['content_id']

dtest = xgb.DMatrix(test_df)
for sub in subjects.values():
    dtrain = xgb.DMatrix(X, Y_all[sub])
    num_rounds = 500
    model = xgb.train(params, dtrain, num_rounds)

    pred = model.predict(dtest)
    result[sub] = pred

result.to_csv('../data/tmp.csv', index=False)

with open('../output/merge1v1.csv', 'w', encoding='utf-8') as f, open('../output/merge1v2.csv', 'w', encoding='utf-8') as f2:
    line = '{},{},0,'
    line2 = '{},{},{},'
    f.write('content_id,subject,sentiment_value,sentiment_word')
    f2.write('content_id,subject,sentiment_value,sentiment_word')
    for index, row in result.iterrows():
        has = False
        for sub in subjects.values():
            if row[sub] != 3:
                has = True
                value = line.format(row['content_id'], sub)
                value2 = line.format(row['content_id'], sub, row[sub]-1)
                f.write('\n')
                f.write(value)
                f2.write('\n')
                f2.write(value2)
        if not has:
            value = line.format(row['content_id'], '价格')
            f.write('\n')
            f.write(value)
            f2.write('\n')
            f2.write(value)



'''