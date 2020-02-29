# -*- coding: utf-8 -*-
"""conet_torch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11nuROYQHAJBedD64gO0zo81qkJEs4LA3
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import sys
w_dir = '/Users/linyishi/Desktop/毕业论文/recommendation_system/CoNet-torch/original_CoNet'
os.chdir(w_dir)
sys.path.append(w_dir)

#!pip install tensorboardX
#!pip install fasttext

import pandas as pd
import numpy as np

from CoNet import CoNetEngine
from data import SampleGenerator
books_df_sample = pd.read_csv('books_df_sample_1000.csv')
movies_df_sample = pd.read_csv('movies_df_sample_1000.csv')

books_df_sample.drop(columns='index',inplace=True)
movies_df_sample.drop(columns='index',inplace=True)

print(books_df_sample['userId'].nunique())
print(books_df_sample['itemId'].nunique())
print(movies_df_sample['itemId'].nunique())
print(books_df_sample.shape)
print(movies_df_sample.shape)

import random
share_u_sample=random.sample(set(books_df_sample['userId']),5000)
books_df_sample = books_df_sample[books_df_sample.userId.isin(set(share_u_sample))].reset_index()
movies_df_sample = movies_df_sample[movies_df_sample.userId.isin(set(share_u_sample))].reset_index()

# 载入元数据和模型
meta_Movies_and_TV_used = pd.read_csv('meta_Movies_and_TV_used.csv',encoding='utf-8-sig')
meta_Books_used = pd.read_csv('meta_Books_used.csv',encoding='utf-8-sig')
import fasttext
fasttext_model = fasttext.load_model("test.bin")

meta_Books_used.columns

# 组合title和description，获取物品的特征表示
# 关于英文文本的预处理，好像在"pytorch自然语言处理"一书中有一参考
def text_preprocessing(text):
  outline = text.replace("\t", " ").replace("\n", " ").replace('.',' ').replace(',',' ').replace('  ',' ')
  outline_list = outline.split(' ')
  outline = " ".join(outline_list)
  return outline
def df_preprocessing(df0):
  assert 'title' in df0.columns,'title不在dataframe的columns里'
  assert 'description' in df0.columns,'description不在dataframe的columns里'
  df = df0.copy()
  df['title']=df['title'].fillna('Unknown')
  df['description']=df['description'].fillna('Unknown')
  df['title&desc'] = df.apply(lambda x:x.title+'\t'+x.description,axis=1)
  df['processd_text'] = df['title&desc'].apply(text_preprocessing)
  return df
meta_Books_used = df_preprocessing(meta_books_used)
#meta_Movies_and_TV_used = df_preprocessing(meta_Movies_and_TV_used)

meta_Books_used.to_csv('meta_Books_used.csv',encoding='utf-8-sig',index=None)
#meta_Movies_and_TV_used.to_csv('meta_Movies_and_TV_used.csv',encoding='utf-8-sig',index=None)

meta_Books_used = pd.read_csv('meta_Books_used.csv')
meta_Movies_and_TV_used = pd.read_csv('meta_Movies_and_TV_used.csv')

#from tqdm import tqdm
#tqdm.pandas(desc="my bar!")
def get_vec(text,model):
  return model.get_sentence_vector(text)
meta_Books_used['vec'] = meta_Books_used['processd_text'].apply(lambda x:get_vec(x,fasttext_model))
meta_Movies_and_TV_used['vec'] = meta_Movies_and_TV_used['processd_text'].apply(lambda x:get_vec(x,fasttext_model))

meta_Movies_and_TV_used.columns

# 匹配元数据
books_df_sample = pd.merge(books_df_sample,meta_Books_used[['asin','vec']],how='left',left_on='itemId',right_on='asin')
movies_df_sample = pd.merge(movies_df_sample,meta_Movies_and_TV_used[['asin','vec']],how='left',left_on='itemId',right_on='asin')

books_df_sample['vec'][0]

#vec_dict_s = books_df_sample[['vec','itemId']].set_index('itemId').to_dict()
mydict = dict(zip(books_df_sample.itemId, books_df_sample.vec))

mydict.keys()

print(len(vec_dict_s))
print(books_df_sample['itemId'].nunique())

from sklearn.preprocessing import LabelEncoder
sparse_features = ['itemId']
dense_features = []

user_lbe = LabelEncoder()
user_lbe.fit(books_df_sample['userId'])
books_df_sample['userId'] = user_lbe.transform(books_df_sample['userId'])
movies_df_sample['userId'] = user_lbe.transform(movies_df_sample['userId'])

for feat in sparse_features:
    lbe = LabelEncoder()
    books_df_sample[feat] = lbe.fit_transform(books_df_sample[feat])

for feat in sparse_features:
    lbe = LabelEncoder()
    movies_df_sample[feat] = lbe.fit_transform(movies_df_sample[feat])
# 这里有问题，两个dataframe对用户ID的encoding不一定一致！！
# 而且，这里至少应该保留原本的itemID以便获取元数据信息，或者直接把元数据的词向量直接贴进去

sample_generator = SampleGenerator(ratings_s=books_df_sample,ratings_t=movies_df_sample)
evaluate_data = sample_generator.evaluate_data

conet_config = {'alias': 'conet_factor8neg2_bz256_1632168_reg_0.0000001',
              'num_epoch': 20,
              'batch_size': 1024,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': books_df_sample['userId'].nunique(),
              'num_items_s': books_df_sample['itemId'].nunique(),
              'num_items_t': movies_df_sample['itemId'].nunique(),
              'device_id': 0,
              'latent_dim': 8,
              'num_negative': 2,
              'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': False,
              'pretrain': False,
              'model_dir':'checkpoints/{}_Epoch{}_HR_s{:.4f}_NDCG_s{:.4f}_HR_t{:.4f}_NDCG_t{:.4f}.model'
                }

config = conet_config
engine = CoNetEngine(config)
train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])

books_df_sample['itemId'].nunique()

import pickle
with open('train_loader.pickle', 'rb') as handle:
    train_loader = pickle.load(handle)

import pickle
with open('train_loader.pickle', 'wb') as handle:
    pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

from importlib import reload # python 2.7 does not require this
import CoNet
reload( CoNet )
from CoNet import CoNetEngine

config = conet_config
engine = CoNetEngine(config)

res=[]
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t = engine.evaluate(evaluate_data, epoch_id=epoch)
    res.append([hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t])
    engine.save(config['alias'], epoch, hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t)

for epoch in range(20,50):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t)

l=[]
for i in engine.model.parameters():
  l.append(i)

len(l)

engine.model