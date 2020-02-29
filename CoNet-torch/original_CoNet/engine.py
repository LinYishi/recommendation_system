import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron_s = MetronAtK(top_k=10)
        self._metron_t = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items_s, ratings_s, items_t, ratings_t):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items_s, ratings_s, items_t, ratings_t = users.cuda(), items_s.cuda(), ratings_s.cuda(),items_t.cuda(), ratings_t.cuda()
        self.opt.zero_grad()
        ratings_pred_s, ratings_pred_t = self.model(users, items_s,items_t)
        loss = self.crit(ratings_pred_s.view(-1), ratings_s) + self.crit(ratings_pred_t.view(-1), ratings_t)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item_s, rating_s, item_t, rating_t = batch[0], batch[1], batch[2], batch[3], batch[4]
            rating_s = rating_s.float()
            rating_t = rating_t.float()
            loss = self.train_single_batch(user, item_s, rating_s, item_t, rating_t)
            if batch_id%1000==0:
                print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items_s, test_items_t = evaluate_data[0], evaluate_data[1],evaluate_data[2]
            negative_users, negative_items_s, negative_items_t = evaluate_data[3], evaluate_data[4], evaluate_data[5]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items_s = test_items_s.cuda()
                test_items_t = test_items_t.cuda()
                negative_users = negative_users.cuda()
                negative_items_s = negative_items_s.cuda()
                negative_items_t = negative_items_t.cuda()
            test_scores_s, test_scores_t = self.model(test_users, test_items_s, test_items_t)
            negative_scores_s, negative_scores_t = self.model(negative_users, negative_items_s, negative_items_t)
            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items_s = test_items_s.cpu()
                test_items_t = test_items_t.cpu()
                negative_users = negative_users.cpu()
                negative_items_s = negative_items_s.cpu()
                negative_items_t = negative_items_t.cpu()
                test_scores_s = test_scores_s.cpu()
                test_scores_t = test_scores_t.cpu()
                negative_scores_s, negative_scores_t = negative_scores_s.cpu(), negative_scores_t.cpu()
            self._metron_s.subjects = [test_users.data.view(-1).tolist(),
                                 test_items_s.data.view(-1).tolist(),
                                 test_scores_s.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items_s.data.view(-1).tolist(),
                                 negative_scores_s.data.view(-1).tolist()]
            self._metron_t.subjects = [test_users.data.view(-1).tolist(),
                                       test_items_t.data.view(-1).tolist(),
                                       test_scores_t.data.view(-1).tolist(),
                                       negative_users.data.view(-1).tolist(),
                                       negative_items_t.data.view(-1).tolist(),
                                       negative_scores_t.data.view(-1).tolist()]
        hit_ratio_s, ndcg_s = self._metron_s.cal_hit_ratio(), self._metron_s.cal_ndcg()
        hit_ratio_t, ndcg_t = self._metron_t.cal_hit_ratio(), self._metron_t.cal_ndcg()
        self._writer.add_scalar('performance/HR_s', hit_ratio_s, epoch_id)
        self._writer.add_scalar('performance/NDCG_s', ndcg_s, epoch_id)
        self._writer.add_scalar('performance/HR_t', hit_ratio_t, epoch_id)
        self._writer.add_scalar('performance/NDCG_t', ndcg_t, epoch_id)
        print('[Evluating Epoch {}] HR_s = {:.4f}, NDCG_s = {:.4f}, HR_t = {:.4f}, NDCG_t = {:.4f}'.format(epoch_id, hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t))
        return hit_ratio_s, ndcg_s, hit_ratio_t, ndcg_t

    def save(self, alias, epoch_id, hit_ratio_s, ndcg_s,hit_ratio_t, ndcg_t):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio_s, ndcg_s,hit_ratio_t, ndcg_t)
        save_checkpoint(self.model, model_dir)