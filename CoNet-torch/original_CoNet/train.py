import os
w_dir = '/Users/linyishi/Desktop/毕业论文/recommendation_system/CoNet-torch/src'
os.chdir(w_dir)
import sys
sys.path.append(w_dir)
import pandas as pd
import numpy as np

from CoNet import CoNetEngine
from data import SampleGenerator

books_df_sample = pd.read_csv('books_df_sample.csv')
movies_df_sample = pd.read_csv('movies_df_sample.csv')

sample_generator = SampleGenerator(ratings_s=books_df_sample,ratings_t=movies_df_sample)
evaluate_data = sample_generator.evaluate_data

os.environ['KMP_DUPLICATE_LIB_OK']='True'


conet_config = {'alias': 'conet_factor8neg4_bz256_166432168_reg_0.0000001',
              'num_epoch': 20,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 1000,
              'num_items_s': 25011,
              'num_items_t': 11756,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': False,
              'pretrain': False,
              'model_dir':'checkpoints/{}_Epoch{}_HR_s{:.4f}_NDCG_s{:.4f}_HR_t{:.4f}_NDCG_t{:.4f}.model'
                }

config = conet_config
engine = CoNetEngine(config)

for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t)

# Load Data
ml1m_dir = 'data/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

useful_columns=['userId','itemId','rating','timestamp']
columns_mapping = {'asin':'itemId','reviewerID':'userId','overall':'rating','unixReviewTime':'timestamp'}

books_df_sample = books_df_sample.rename(columns=columns_mapping)
movies_df_sample = movies_df_sample.rename(columns=columns_mapping)
books_df_sample = books_df_sample[useful_columns]
movies_df_sample = movies_df_sample[useful_columns]

# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
# DataLoader for training
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data
# Specify the exact model
#config = gmf_config
#engine = GMFEngine(config)
config = mlp_config
engine = MLPEngine(config)
# config = neumf_config
# engine = NeuMFEngine(config)
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)