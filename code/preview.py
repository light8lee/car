
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import jieba


# In[2]:


df = pd.read_csv('../data/train.csv')
df.head(5)


# In[3]:


df.shape


# In[4]:


df[df.subject=='价格'].shape


# In[5]:


df.groupby('subject').describe()


# ### 从词向量文件中，提取需要的词的向量，减少大小

# In[6]:


word_set = set()


# In[7]:


for line in df['content']:
    words = jieba.cut(line)
    word_set.update(words)


# In[10]:


in_count = 0
with open('../data/merge_sgns_bigram_char300.txt', encoding='utf-8') as f, open('../data/cliped_vec.vec', 'w', encoding='utf-8') as outf:
    f.readline()
    while True:
        line = f.readline()
        if not line:
            break
        try:
            word, vec = line.split(' ', 1)
        except:
            print(line)
            continue
        if word in word_set:
            in_count += 1
            outf.write(line)
            outf.write('/n')
        else:
            print(word)
print('total have ', in_count, 'words in word embedding')


# In[9]:


len(word_set)

