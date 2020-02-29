import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from collections import Counter

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor_s, target_tensor_s, item_tensor_t, target_tensor_t, item_vecs_s, item_vecs_t):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor_s = item_tensor_s
        self.target_tensor_s = target_tensor_s
        self.item_tensor_t = item_tensor_t
        self.target_tensor_t = target_tensor_t
        self.item_vecs_s = item_vecs_s
        self.item_vecs_t = item_vecs_t

    def __getitem__(self, index):
            return self.user_tensor[index], self.item_tensor_s[index], self.target_tensor_s[index],self.item_tensor_t[index], self.target_tensor_t[index], self.item_vecs_s[index], self.item_vecs_t[index]


    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings_s,ratings_t):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings_s.columns
        assert 'itemId' in ratings_s.columns
        assert 'rating' in ratings_s.columns
        assert 'userId' in ratings_t.columns
        assert 'itemId' in ratings_t.columns
        assert 'rating' in ratings_t.columns
        assert ratings_s['userId'].nunique()==ratings_t['userId'].nunique()
        self.ratings_s = ratings_s
        self.ratings_t = ratings_t
        # 获取物品的itemId和词向量的字典
        self.vec_dict_s = dict(zip(ratings_s.itemId, ratings_s.vec))
        self.vec_dict_t = dict(zip(ratings_t.itemId, ratings_t.vec))

        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings_s = self._binarize(ratings_s)
        self.preprocess_ratings_t = self._binarize(ratings_t)
        self.user_pool = set(self.ratings_s['userId'].unique())
        self.item_pool_s = set(self.ratings_s['itemId'].unique())
        self.item_pool_t = set(self.ratings_t['itemId'].unique())
        self.item_pool_s_top1w=set([x[0] for x in Counter(self.ratings_s['itemId']).most_common(10000)])
        self.item_pool_t_top1w = set([x[0] for x in Counter(self.ratings_t['itemId']).most_common(10000)])
        # create negative item samples for NCF learning
        self.negatives_s = self._sample_negative(ratings_s,'s').rename(columns={'negative_items':'negative_items_s','negative_samples':'negative_samples_s'})
        self.negatives_t = self._sample_negative(ratings_t,'t').rename(columns={'negative_items':'negative_items_t','negative_samples':'negative_samples_t'})
        self.train_ratings_s, self.test_ratings_s = self._split_loo(self.preprocess_ratings_s)
        self.train_ratings_t, self.test_ratings_t = self._split_loo(self.preprocess_ratings_t)
        """pair source domain items with target domain items"""
        self.merged_train = pd.merge(self.train_ratings_s, self.train_ratings_t, on=['userId','rank_latest'], how='inner', suffixes=('_s', '_t'))
        self.merged_test = pd.merge(self.test_ratings_s, self.test_ratings_t, on='userId', suffixes=('_s', '_t'))
        self.merged_train.drop(columns=['rank_latest'],inplace=True)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating','rank_latest', 'vec']], test[['userId', 'itemId', 'rating', 'vec']]

    def _sample_negative(self, ratings,domain):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        # 就是这一步，特别耗时耗内存
        if domain=='s':
            interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool_s_top1w - x)
        else:
            interact_status['negative_items'] = interact_status['interacted_items'].apply(
                lambda x: self.item_pool_t_top1w - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]



    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items_s, ratings_s, items_t, ratings_t = [], [], [], [], []
        # 增加了物品的词向量表示
        item_vecs_s, item_vecs_t = [], []
        train_ratings = pd.merge(self.merged_train, self.negatives_s[['userId', 'negative_items_s']], on='userId')
        train_ratings = pd.merge(train_ratings, self.negatives_t[['userId', 'negative_items_t']], on='userId',
                                 suffixes=('', '_t'))
        train_ratings['negatives_s'] = train_ratings['negative_items_s'].apply(lambda x: random.sample(x, num_negatives))
        train_ratings['negatives_t'] = train_ratings['negative_items_t'].apply(
            lambda x: random.sample(x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items_s.append(int(row.itemId_s))
            ratings_s.append(float(row.rating_s))
            items_t.append(int(row.itemId_t))
            ratings_t.append(float(row.rating_t))
            #
            item_vecs_s.append(row.vec_s)
            item_vecs_t.append(row.vec_t)
            for i in range(num_negatives):
                users.append(int(row.userId))
                items_s.append(int(row.negatives_s[i]))
                ratings_s.append(float(0))  # negative samples get 0 rating
                items_t.append(int(row.negatives_t[i]))
                ratings_t.append(float(0))  # negative samples get 0 rating
                #
                item_vecs_s.append(self.vec_dict_s[row.negatives_t[i]])
                item_vecs_t.append(self.vec_dict_t[row.negatives_t[i]])

        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor_s=torch.LongTensor(items_s),
                                        target_tensor_s=torch.FloatTensor(ratings_s),
                                        item_tensor_t=torch.LongTensor(items_t),
                                        target_tensor_t=torch.FloatTensor(ratings_t),
                                        item_vecs_s = torch.Tensor(item_vecs_s),
                                        item_vecs_t=torch.Tensor(item_vecs_t)
                                        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_ratings = pd.merge(self.merged_test, self.negatives_s[['userId', 'negative_samples_s']], on='userId')
        test_ratings = pd.merge(test_ratings, self.negatives_t[['userId', 'negative_samples_t']], on='userId')
        test_users, test_items_s, test_items_t, negative_users, negative_items_s, negative_items_t = [], [], [], [], [], []
        #
        item_vecs_s, item_vecs_t, neg_item_vecs_s, neg_item_vecs_t = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items_s.append(int(row.itemId_s))
            test_items_t.append(int(row.itemId_t))
            #
            item_vecs_s.append(row.vec_s)
            item_vecs_t.append(row.vec_t)
            for i in range(len(row.negative_samples_s)):
                negative_users.append(int(row.userId))
                negative_items_s.append(int(row.negative_samples_s[i]))
                negative_items_t.append(int(row.negative_samples_t[i]))
                #
                neg_item_vecs_s.append(self.vec_dict_s[row.negative_samples_s[i]])
                neg_item_vecs_t.append(self.vec_dict_t[row.negative_samples_t[i]])

        return [torch.LongTensor(test_users), torch.LongTensor(test_items_s), torch.LongTensor(test_items_t), torch.Tensor(item_vecs_s), torch.Tensor(item_vecs_t),
                torch.LongTensor(negative_users),torch.LongTensor(negative_items_s), torch.LongTensor(negative_items_t),
                torch.Tensor(neg_item_vecs_s), torch.Tensor(neg_item_vecs_t)]
