
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import lightgbm as lgb
from numpy import loadtxt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import argparse
from params_config import LGBConfig as Config

parser = argparse.ArgumentParser()
# parser.add_argument('--num_round', type=int, default=500, help='训练次数')
parser.add_argument('--cv', dest='cv_flag', action='store_true', help='进行交叉验证, 默认')
parser.add_argument('--nocv', dest='cv_flag', action='store_false', help='不进行交叉验证')
parser.set_defaults(cv_flag=True)
parser.add_argument('--pred', dest='pred', action='store_true', help='进行预测')
parser.add_argument('--nopred', dest='pred', action='store_false', help='不进行预测，默认')
parser.set_defaults(pred=False)
parser.add_argument('--kfold', type=int, default=5, help='交叉验证的ｋ折数，默认５折')
parser.set_defaults(binary=False)
args = parser.parse_args()


Y_all = pd.read_csv('../data/input2.csv')



seed = 11

meta_params = {
    'objective': 'binary',    # 多分类的问题
    # 'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    # 'max_depth': 12,               # 构建树的深度，越大越容易过拟合
    # 'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    # 'subsample': args.subsample,              # 随机采样训练样本
    # 'colsample_bytree': args.col_subsample,       # 生成树时进行的列采样
    # 'min_child_weight': 3,
    # 'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    # 'eta': args.eta,                  # 如同学习率
    'seed': seed,
    # 'nthread': 4,                  # cpu 线程数
}

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

weights = {
    '价格': 5.512175962293794,
    '内饰': 14.466417910447761,
    '动力': 2.034407027818448,
    '外观': 15.952965235173824,
    '安全性': 13.467713787085515,
    '操控': 7.001930501930502,
    '油耗': 6.66173752310536,
    '空间': 17.755656108597286,
    '舒适性': 7.904403866809882,
    '配置': 8.718640093786636,
}

def get_df(Xs, data_name):
    if data_name not in Xs:
        df = pd.read_csv('../data/{}.csv'.format(data_name))
        Xs[data_name] = df
    return Xs[data_name]

Xs = dict()

if args.cv_flag:
    print('cross validating...')
    history = []
    avg_f1 = 0
    for name, sub in subjects.items():
        params, dname = Config[name]
        data_name = '{}_train'.format(dname)
        params['scale_pos_weight'] = weights[sub] # 处理不平衡数据
        params.update(meta_params)
        model = lgb.LGBMClassifier(**params)

        scores = cross_val_score(model, get_df(Xs, data_name), Y_all[sub], cv=5, scoring='f1')
        history.append((sub, scores.mean()))
        avg_f1 += scores.mean()
        print(sub, 'done.')
    print(history)
    print(avg_f1/10)



if args.pred:
    # test_df = pd.read_csv('../data/{}_test.csv'.format(args.data_name))
    result = pd.DataFrame()
    test = pd.read_csv('../data/test_public.csv')
    result['content_id'] = test['content_id']

    for name, sub in subjects.items():
        params, dname = Config[name]

        params['scale_pos_weight'] = weights[sub] # 处理不平衡数据
        params.update(meta_params)

        model = lgb.LGBMClassifier(**params)
        train_name = '{}_train'.format(dname)
        model.fit(get_df(Xs, train_name), Y_all[sub])
        test_name = '{}_test'.format(dname)
        pred = model.predict_proba(get_df(Xs, test_name))
        proba_df = pd.DataFrame(pred, columns=['neg_proba', 'pos_proba'])
        proba_df.to_csv('../tmp/lgb-{}.csv'.format(name), index=False)
        print(sub, 'done.')
