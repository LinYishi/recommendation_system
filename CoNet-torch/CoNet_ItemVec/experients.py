from google.colab import drive
drive.mount('/content/drive')

import os
import sys
w_dir = '/content/drive/My Drive/Colab Notebooks/CoNet-torch/src'
os.chdir(w_dir)
sys.path.append(w_dir)

import pandas as pd
import numpy as np
import torch
import random
from sklearn.preprocessing import LabelEncoder
import fasttext
import pickle

from CoNet import CoNetEngine
from data import SampleGenerator
books_df = pd.read_csv('books_df_sample.csv')
movies_df = pd.read_csv('movies_df_sample.csv')
books_df.drop(columns='index',inplace=True)
movies_df.drop(columns='index',inplace=True)


def sample_data(df1,df2,n):
    '''
    采样数据
    '''
    assert "userId" in df1.columns
    assert df1["userId"].nunique()==df2['userId'].unique()
    share_u_sample = random.sample(set(df1['userId']),n)
    df1_sample = df1[df1.userId.isin(set(share_u_sample))].reset_index()
    df2_sample = df2[df2.userId.isin(set(share_u_sample))].reset_index()
    return df1_sample, df2_sample

def lbencoder(df1,df2,sparse_features=['itemId']):
    '''
    对ID类特征进行编码
    '''
    assert 'userId' in df1.columns
    assert 'userId' in df2.columns
    assert df1['userId'].nunique()==df2['userId'].nunique()
    for col in sparse_features:
        assert col in df1.columns
        assert col in df2.columns
    user_lbe = LabelEncoder()
    user_lbe.fit(df1['userId'])
    df1['userId'] = user_lbe.transform(df1['userId'])
    df2['userId'] = user_lbe.transform(df2['userId'])
    for feat in sparse_features:
        lbe = LabelEncoder()
        df1[feat] = lbe.fit_transform(df1[feat])
    for feat in sparse_features:
        lbe = LabelEncoder()
        df2[feat] = lbe.fit_transform(df2[feat])
    return df1,df2


def get_vec(text, model):
    return model.get_sentence_vector(text)

def remove_nan_vec(df,col='vec'):
  nan_indexs = []
  for i,row in df.iterrows():
    if not row[col][0]==row[col][0]:
      print(i)
      nan_indexs.append(i)
  if len(nan_indexs)==0:
    return df
  mean_vec = df.loc[0,col]
  for i in range(1,len(df)):
    v = df.loc[i,col]
    if v[0] == v[0]:
      mean_vec = mean_vec + v
  mean_vec = mean_vec/len(df)
  for idx in nan_indexs:
    df.set_value(idx,col,mean_vec)
    print('fill {}'.format(idx))
  return df

def assign_vec(df1,df2,itemVec_file):
    '''
    匹配上嵌入特征,必须在lbe之前！！
    :param df1:
    :param df2:
    :param itemVec_file:path
    :return:
    '''
    meta_Movies_and_TV_used = pd.read_pickle("meta_Movies_and_TV_used_vec.pkl")
    meta_Books_used = pd.read_pickle("meta_Books_used_vec.pkl")
    if itemVec_file.endswith('bin'):
        fasttext_model = fasttext.load_model(itemVec_file)
        meta_Books_used['vec'] = meta_Books_used['processd_text'].apply(lambda x: get_vec(x, fasttext_model))
        meta_Movies_and_TV_used['vec'] = meta_Movies_and_TV_used['processd_text'].apply(
            lambda x: get_vec(x, fasttext_model))
    elif itemVec_file.endswith('pkl'):
        meta_Movies_and_TV_used = remove_nan_vec(meta_Movies_and_TV_used)
        meta_Books_used = remove_nan_vec(meta_Books_used)
    df1 = pd.merge(df1, meta_Books_used[['asin', 'vec']], how='left', left_on='itemId',
                               right_on='asin')
    df2 = pd.merge(df2, meta_Movies_and_TV_used[['asin', 'vec']], how='left',
                                left_on='itemId', right_on='asin')
    return df1,df2


use_cuda = torch.cuda.is_available()
latent_dim = 8
itemVec_dim = 16
layers = [2*latent_dim+itemVec_dim, 32,16,8]
params = {'books_df':books_df,
          'movies_df':movies_df,
          'latent_dim':latent_dim,
          'num_negative':2,
          'use_cuda':use_cuda,
          'layers':layers,
          'sample_size':5000,
          'epoch':20,
          'batch_size':256,
          'id':0,
          'use_itemVec':False,
          'itemVec_file':None,
          'weight_decay':0.01
          }

def main(params):
    books_df_sample, movies_df_sample = sample_data(params['books_df'], params['movies_df'], params['sample_size'])
    if params['use_itemVec']:
        books_df_sample, movies_df_sample = assign_vec(books_df_sample, movies_df_sample, params['itemVec_file'])
    books_df_sample, movies_df_sample = lbencoder(books_df_sample, movies_df_sample, 5000)
    sample_generator = SampleGenerator(ratings_s=books_df_sample, ratings_t=movies_df_sample)
    evaluate_data = sample_generator.evaluate_data
    alias = 'conetItemVecc_factor{}neg{}_bz{}_{}_reg_0.0000001_{}'.format(\
        params['latent_dim'],params['num_negative'],params['batch_size'],''.join(params['layers']),params['id'])
    config = {'alias': alias,
                    'num_epoch': params['epoch'],
                    'batch_size': params['batch_size'],
                    'optimizer': 'adam',
                    'adam_lr': 1e-3,
                    'num_users': books_df_sample['userId'].nunique(),
                    'num_items_s': books_df_sample['itemId'].nunique(),
                    'num_items_t': movies_df_sample['itemId'].nunique(),
                    'device_id': 0,
                    'latent_dim': params['latent_dim'],
                    'num_negative': params['num_negative'],
                    'layers': params['layers'],  # layers[0] is the concat of latent user vector & latent item vector
                    'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
                    'use_cuda': params['use_cuda'],
                    'pretrain': False,
                    'model_dir': 'checkpoints/{}_Epoch{}_HR_s{:.4f}_NDCG_s{:.4f}_HR_t{:.4f}_NDCG_t{:.4f}.model'
                    }
    engine = CoNetEngine(config)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    res = []
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t = engine.evaluate(evaluate_data, epoch_id=epoch)
        res.append([hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t])
        engine.save(config['alias'], epoch, hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t)
    return res

