import jieba
import pandas as pd
import numpy as np
from collections import defaultdict
import re

train_df = pd.read_csv('../data/input2.csv')
test_df = pd.read_csv('../data/test_public.csv')

corpus = train_df['content']

word_count = defaultdict(int)
for doc in corpus:
    for w in jieba.cut(doc):
        word_count[w] += 1

punc_ptn = re.compile(r"[\s+\/_,$%^*(+\"\')]+|[:：+——()?【】“”，。、~@#￥%……&*（）]+")
valid_words = []
with open('../data/bow_words.txt', 'w', encoding='utf-8') as f:
    for w, count in word_count.items():
        if not punc_ptn.search(w) and count >= 5:
            valid_words.append(w)
            f.write(w)
            f.write('\n')

print(len(valid_words))

# 生成训练数据和测试数据
tfidf_train = pd.DataFrame()
tfidf_test = pd.DataFrame()
for w in valid_words:
    tfidf_train[w] = train_df['content'].apply(lambda text: 1 if text.find(w) != -1 else 0).astype(np.int32)
    tfidf_test[w] = test_df['content'].apply(lambda text: 1 if text.find(w) != -1 else 0).astype(np.int32)

print(tfidf_train.shape)
print(tfidf_test.shape)

tfidf_train.to_csv('../data/bowna_train.csv', index=False)
tfidf_test.to_csv('../data/bowna_test.csv', index=False)
