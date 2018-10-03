
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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_name', help='输入的文件名')
parser.add_argument('--out_name', default='out', help="输出的文件名，当使用`--pred`时有效")
parser.add_argument('--num_round', type=int, default=500, help='训练次数')
parser.add_argument('--cv', dest='cv_flag', action='store_true', help='进行交叉验证, 默认')
parser.add_argument('--nocv', dest='cv_flag', action='store_false', help='不进行交叉验证')
parser.set_defaults(cv_flag=True)
parser.add_argument('--pred', dest='pred', action='store_true', help='进行预测')
parser.add_argument('--nopred', dest='pred', action='store_false', help='不进行预测，默认')
parser.set_defaults(pred=False)
parser.add_argument('--kfold', type=int, default=5, help='交叉验证的ｋ折数，默认５折')
parser.add_argument('--binary', dest='binary', action='store_true', help='进行二分类')
parser.add_argument('--nobinary', dest='binary', action='store_false', help='不进行二分类，而使用四分类，默认')
parser.set_defaults(binary=False)
parser.add_argument('--col_subsample', type=float, default=1.0, help="列采样率")
parser.add_argument('--subsample', type=float, default=1.0, help="样本采样率")
parser.add_argument('--eta', type=float, default=0.3, help='学习率')
args = parser.parse_args()


# In[ ]:


X = pd.read_csv('../data/{}_train.csv'.format(args.data_name))


# In[ ]:


if args.binary:
    Y_all = pd.read_csv('../data/input2.csv')
else:
    Y_all = pd.read_csv('../data/input4.csv')


# ### 验证

# In[ ]:


# In[ ]:


seed = 7
# test_size = 1000
# X_train, X_test, y_train, y_test = train_test_split(X, Y_all, test_size=test_size, random_state=seed)


# In[ ]:


params = {
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'objective': 'multi:softmax',    # 多分类的问题
    # 'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    # 'max_depth': 12,               # 构建树的深度，越大越容易过拟合
    # 'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': args.subsample,              # 随机采样训练样本
    'colsample_bytree': args.col_subsample,       # 生成树时进行的列采样
    # 'min_child_weight': 3,
    # 'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': args.eta,                  # 如同学习率
    'seed': seed,
    # 'nthread': 4,                  # cpu 线程数
}

if args.binary:
    params['num_class'] = 2 # 类别数，与 multisoftmax 并用
else:
    params['num_class'] = 4 # 类别数，与 multisoftmax 并用


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

def f1_eval(preds, dtrain):
    labels = dtrain.get_label()
    score = f1_score(labels, preds, average='macro')
    return 'f1_eval', score

if args.cv_flag:
    print('cross validating...')
    history = []
    avg_f1 = 0
    for sub in subjects.values():
        dtrain = xgb.DMatrix(X, Y_all[sub])
        num_round = args.num_round
        if args.binary: # 处理不平衡数据
            params['scale_pos_weight'] = weights[sub]
        result = xgb.cv(params, dtrain, num_round, nfold=args.kfold, maximize=True, feval=f1_eval, shuffle=True)
        test_eval_mean = result.loc[num_round-1, 'test-f1_eval-mean']
        train_eval_mean = result.loc[num_round-1, 'train-f1_eval-mean']
        history.append((sub, test_eval_mean, train_eval_mean))
        avg_f1 += test_eval_mean
    print(history)
    print(avg_f1)



if args.pred:
    test_df = pd.read_csv('../data/{}_test.csv'.format(args.data_name))
    result = pd.DataFrame()
    test = pd.read_csv('../data/test_public.csv')
    result['content_id'] = test['content_id']

    dtest = xgb.DMatrix(test_df)
    for sub in subjects.values():
        dtrain = xgb.DMatrix(X, Y_all[sub])
        num_rounds = args.num_round
        if args.binary: # 处理不平衡数据
            params['scale_pos_weight'] = weights[sub]
        model = xgb.train(params, dtrain, num_rounds)

        pred = model.predict(dtest)
        result[sub] = pred

    result.to_csv('../data/tmp.csv', index=False)


    with open('../output/{}v1.csv'.format(args.out_name), 'w', encoding='utf-8') as f, open('../output/{}v2.csv'.format(args.out_name), 'w', encoding='utf-8') as f2:
        line = '{},{},0,'
        line2 = '{},{},{},'
        f.write('content_id,subject,sentiment_value,sentiment_word')
        f2.write('content_id,subject,sentiment_value,sentiment_word')
        for index, row in result.iterrows():
            has = False
            for sub in subjects.values():
                if args.binary: # 二分类情况，预测结果为１表示存在该主题，情感填０
                    if row[sub]:
                        has = True
                        value = line.format(row['content_id'], sub)
                        f.write('\n')
                        f.write(value)
                else: # 四分类情况，预测结果不为３表示存在主题，情感为类别-1
                    if row[sub] != 3:
                        has = True
                        value = line.format(row['content_id'], sub)
                        value2 = line.format(row['content_id'], sub, row[sub]-1)
                        f.write('\n')
                        f.write(value)
                        f2.write('\n')
                        f2.write(value2)
            if not has: # 没有主题，默认值为价格主题，情感为０
                value = line.format(row['content_id'], '价格')
                f.write('\n')
                f.write(value)
                f2.write('\n')
                f2.write(value)



