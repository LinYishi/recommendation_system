import torch
from gmf import GMF
from engine import Engine
from utils import use_cuda, resume_checkpoint


class CoNet_model(torch.nn.Module):
    def __init__(self, config):
        super(CoNet_model, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items_s = config['num_items_s']
        self.num_items_t = config['num_items_t']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item_s = torch.nn.Embedding(num_embeddings=self.num_items_s, embedding_dim=self.latent_dim)
        self.embedding_item_t = torch.nn.Embedding(num_embeddings=self.num_items_t, embedding_dim=self.latent_dim)

        self.fc_layers_s = torch.nn.ModuleList()
        self.fc_layers_t = torch.nn.ModuleList()
        self.transfer_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers_s.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers_t.append(torch.nn.Linear(in_size, out_size))
            self.transfer_layers.append(torch.nn.Linear(in_size, out_size))
        #if self.config['use_cuda'] is True:
        #    for i in range(len(self.transfer_matrix)):
        #        self.transfer_matrix[i] = self.transfer_matrix[i].cuda()
        self.affine_output_s = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.affine_output_t = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices_s,item_indices_t):
        user_embedding = self.embedding_user(user_indices)
        item_embedding_s = self.embedding_item_s(item_indices_s)
        item_embedding_t = self.embedding_item_t(item_indices_t)

        vector_s = torch.cat([user_embedding, item_embedding_s], dim=-1)  # the concat latent vector
        vector_t = torch.cat([user_embedding, item_embedding_t], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers_s))):

            #vector_s,vector_t = self.fc_layers[idx](vector_s) + torch.matmul(vector_t,self.transfer_matrix[idx]),self.fc_layers[idx](vector_t) + torch.matmul(vector_s,self.transfer_matrix[idx])
            vector_s, vector_t = self.fc_layers_s[idx](vector_s) + self.transfer_layers[idx](vector_t), \
                                 self.fc_layers_t[idx](vector_t) + self.transfer_layers[idx](vector_s)
            vector_s,vector_t = torch.nn.ReLU()(vector_s),torch.nn.ReLU()(vector_t)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        logits_s = self.affine_output_s(vector_s)
        logits_t = self.affine_output_t(vector_t)
        rating_s = self.logistic(logits_s)
        rating_t = self.logistic(logits_t)
        return rating_s,rating_t

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        config = self.config
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
        self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item.weight.data = gmf_model.embedding_item.weight.data


class CoNetEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = CoNet_model(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(CoNetEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()