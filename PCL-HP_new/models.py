import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np
import math
import random


class DHP(nn.Module):
    def __init__(self, nodes, nfeat, nhid1, noutput, dropout, n_feature, n_hidden, n_output, hidden_size, loss_decay, seed, ratio, loss_temp):
        super(DHP, self).__init__()
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gc1 = GraphConvolution(nfeat, nhid1, seed)
        self.gc2 = GraphConvolution(nhid1, noutput, seed)

        self.hgc1 = GraphConvolution(nfeat, nhid1, seed)
        self.hgc2 = GraphConvolution(nhid1, noutput, seed)

        self.dropout = dropout
        self.sm = nn.Sigmoid()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

        self.hidden2 = torch.nn.Linear(2 * n_feature, n_hidden)
        self.predict2 = torch.nn.Linear(n_hidden, n_output)

        self.sm = nn.Sigmoid()
        self.initial_state_matrix1 = torch.nn.Embedding(nodes, nfeat)
        self.initial_state_matrix2 = torch.nn.Embedding(nodes, nfeat)

        self.project = nn.Sequential(nn.Linear(noutput, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1, bias=False))

        self.ratio = ratio
        self.seed = seed
        self.loss_decay = loss_decay
        self.loss_temp = loss_temp

    def forward_higher(self, adj_matrix):
        x0 = self.initial_state_matrix1.weight
        x1 = self.hgc1(x0, adj_matrix)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = (self.hgc2(x1, adj_matrix))
        return x2

    def forward_lower(self, adj_matrix, pos_sample, neg_sample):
        x0 = self.initial_state_matrix2.weight
        layer_loss = self.compute_inner_prob(x0, pos_sample, neg_sample)
        x1 = self.gc1(x0, adj_matrix)
        layer_loss = layer_loss + self.compute_inner_prob(x1, pos_sample, neg_sample)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = (self.gc2(x1, adj_matrix))
        layer_loss = layer_loss + self.compute_inner_prob(x2, pos_sample, neg_sample)
        return x0 + (1/2) * x1 + (1/3) * x2, layer_loss

    def forward3(self, x):
        x0 = F.relu(self.hidden(x))
        x1 = self.predict(x0)
        return x1

    def forward4(self, x):
        x0 = F.relu(self.hidden2(x))
        x1 = self.predict2(x0)
        return x1

    def attention(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        x = (beta * z)
        return x.sum(1), beta

    def dynamics(self, input_matrix, degree, lap_p, lap_n):
        trans_degree = F.softmax(torch.tensor(degree)).resize(input_matrix.shape[0], 1)#* trans_degree
        a = math.pi * trans_degree * (torch.ones(input_matrix.shape[0], input_matrix.shape[1]).to(self.device))
        output_matrix1 = torch.sin(torch.mm(torch.tensor(lap_p), input_matrix.to(torch.float64) - a))  #
        output_matrix2 = torch.sin(torch.mm(torch.tensor(lap_n), input_matrix.to(torch.float64) - a))  #
        return output_matrix1, output_matrix2

    def compute_inner_prob(self, embedding, pos, neg):
        feature_pooling, order_pooling = pooling_method1(self.ratio, embedding, pos, neg, len(pos))
        order_embedding, hyperedge_labels = perturb_feature(self.seed, feature_pooling, order_pooling)
        MLP_prob = self.forward3(order_embedding.to(torch.float32))
        epoch_loss = self.compute_loss(MLP_prob, hyperedge_labels.to(self.device), 0)
        return epoch_loss

    def compute_outside_prob(self, embedding, pos, neg):
        feature_pooling, order_pooling = pooling_method1(self.ratio, embedding, pos, neg, len(pos))
        order_embedding, hyperedge_labels = perturb_feature(self.seed, feature_pooling, order_pooling)
        MLP_prob = self.forward4(order_embedding.to(torch.float32))
        epoch_loss = self.compute_loss(MLP_prob, hyperedge_labels.to(self.device), 0)
        return epoch_loss

    def compute_loss(self, MLP_output, order_labels, c1):
        loss1 = F.binary_cross_entropy_with_logits(MLP_output, order_labels)
        loss = self.loss_decay * loss1
        return loss

    def calculate_con_loss(self, feature):
        random.seed(self.seed)
        shuffle_indices = np.random.permutation(feature.shape[0])
        f_shuffled = feature[shuffle_indices]
        con_loss1 = self.contrastive_loss(feature, f_shuffled, self.loss_temp)
        con_loss2 = self.contrastive_loss(f_shuffled, feature, self.loss_temp)
        con_loss = (con_loss2 + con_loss1) * 0.5
        con_loss = con_loss.mean()
        return con_loss

    def calculate_hl_loss(self, feature1, feature2):
        hl_loss1 = self.contrastive_loss(feature1, feature2, self.loss_temp)
        hl_loss2 = self.contrastive_loss(feature2, feature1, self.loss_temp)
        hl_loss = (hl_loss2 + hl_loss1) * 0.5
        hl_loss = hl_loss.mean()
        return hl_loss

    def contrastive_loss(self, A, B, te):
        between_sim = self.f(self.cosine_similarity(A, B), te)
        return -torch.log(between_sim.diag() / between_sim.sum(1))

    def cosine_similarity(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def f(self, x, tau):
        return torch.exp(x / tau)


def pooling_method1(ratio, node_feature, pos, neg, simp_num):
    for i in range(simp_num):
        s = node_feature[pos[i]]
        pos_vector = torch.mean(s, dim=0).reshape(1, node_feature.shape[1])
        ratio_neg = []
        for j in range(ratio):
            s2 = node_feature[neg[i * ratio + j]]
            neg_vector = torch.mean(s2, dim=0).reshape(1, node_feature.shape[1])
            ratio_neg.append(neg_vector)
        if i == 0:
            pos_feature = pos_vector
            neg_feature = torch.stack(ratio_neg).reshape(ratio, node_feature.shape[1])
        else:
            pos_feature = torch.cat((pos_feature, pos_vector), 0)
            neg_feature = torch.cat((neg_feature, torch.stack(ratio_neg).reshape(ratio, node_feature.shape[1])), 0)
    labels = torch.cat((torch.ones(len(pos)).reshape(1, len(pos)), torch.zeros(len(neg)).reshape(1, len(neg))), 1)
    return torch.cat((pos_feature, neg_feature), 0), labels.T


def perturb_feature(seed_number, feature_matrix, labels):
    random.seed(seed_number)
    new_raw = np.random.permutation(feature_matrix.shape[0])
    final_features = feature_matrix[new_raw, :]
    final_labels = labels[new_raw, :]
    return final_features, final_labels