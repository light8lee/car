import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('subject')
args = parser.parse_args()

X = pd.read_csv('../data/merge_train.csv')
X.head(1)


Y_all = pd.read_csv('../data/input.csv')
Y_all.head(1)


seed = 7
test_size = 1000
X_train, X_test, y_train, y_test = train_test_split(
    X, Y_all[args.subject], test_size=test_size, random_state=seed)

param = {  # 慢慢来,参数太多怕时间会炸
    # 参考: https://yyqing.me/post/2017/2017-10-23-xgboost-tune
    # round 1
    'learning_rate': [0.01, 0.05, 0.08, 0.1],
    'booster': ['gbtree', 'dart'],
    'n_estimators': [100, 120],

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

model = XGBClassifier(tree_method='gpu_hist')

eval_set = [(X_train, y_train), (X_test, y_test)]

kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
grid_search = GridSearchCV(model, param, scoring='f1_micro', n_jobs=-1, cv=kfold)
print('开始对 {} 进行调参...'.format(args.subject))
grid_result = grid_search.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=12)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
