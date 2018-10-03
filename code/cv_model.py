import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import itertools
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('subject')
parser.add_argument('--binary', dest='binary', action='store_true', help='进行二分类')
parser.add_argument('--nobinary', dest='binary', action='store_false', help='不进行二分类，而使用四分类，默认')
parser.add_argument('--kfold', type=int, default=5, help='交叉验证的ｋ折数，默认５折')
parser.set_defaults(binary=False)
args = parser.parse_args()

X = pd.read_csv('../data/merge_train.csv')


if args.binary:
    Y_all = pd.read_csv('../data/input2.csv')
else:
    Y_all = pd.read_csv('../data/input4.csv')
Y = Y_all[args.subject]


seed = 7

num_rounds = [400, 500, 600]
params = {  # 慢慢来,参数太多怕时间会炸
    # 参考: https://yyqing.me/post/2017/2017-10-23-xgboost-tune
    # round 1
    'learning_rate': [0.05, 0.08, 0.1],

    # round 2
    # 'min_child_weight': [1, 2, 3],
    # 'max_depth': [6, 8, 10],

    # round 3
    # 'gamma': [0.01, 0.05, 0.1, 0.12],

    # round 4
    # 'subsample': [0.5, 0.6, 0.7, 0.8],

    # round 5
    # 'reg_lambda': [1, 1.5, 2],
    # 'reg_alpha': [0.0008, 0.01, 0.05, 0.08, 0.1],

    # round 5
    # 'n_estimators': [], # 比之前的增加
    # 'learning_rate': [], # 比之前的减少
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

meta_param = {
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'seed': seed,
    'objective': 'multi:softmax',    # 多分类的问题
}
if args.binary:
    meta_param['num_class'] = 2 # 类别数，与 multisoftmax 并用
else:
    meta_param['num_class'] = 4 # 类别数，与 multisoftmax 并用

print('cross validating...')
history = []
best = None
best_score = 0

list_params_keys = list(params.keys())
list_params_values = [params[key] for key in list_params_keys]

def print_status(ps, num_round, test_eval_mean):
    for key, value in zip(list_params_keys, ps):
        print(key, '=', value, ',', end='')
    print('round =', num_round, ': ', test_eval_mean)

dtrain = xgb.DMatrix(X, Y)
for num_round in num_rounds:
    for ps in itertools.product(*list_params_values):
        param = {key: value for key, value in zip(list_params_keys, ps)}
        if args.binary: # 处理不平衡数据
            param['scale_pos_weight'] = weights[args.subject]
        result = xgb.cv(params, dtrain, num_round, nfold=args.kfold, maximize=True, feval=f1_eval, shuffle=True)
        test_eval_mean = result.loc[num_round-1, 'test-f1_eval-mean']
        train_eval_mean = result.loc[num_round-1, 'train-f1_eval-mean']
        print_status(ps, num_round, test_eval_mean)
        if test_eval_mean > best_score:
            best = (ps, num_round)
            best_score = test_eval_mean
            print('New Record!')

print('best score:', best_score)
print_status(best[0], best[1], best)