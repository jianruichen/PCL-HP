import numpy as np
import random
import scipy.sparse as sp
import torch
import math


def preprocess_adj_pos_neg(ratio, seed_num, node_number, window_num, data, high_data):
    data = np.array(data).reshape(len(data), 3)
    new_time_list = data[:, -1]
    high_new_time_list = [high_data[i][0, -1] for i in range(len(high_data))]
    time_unique = np.unique(high_new_time_list)
    per_scale = np.round(len(time_unique) / window_num)
    adj_list = []
    pos = []
    neg = []
    degree = []
    lap_p = []
    lap_n = []
    clus = []
    for i in range(window_num):
        if i == 0:
            down_raw = time_unique[int(per_scale * (i + 1) - 1)]
            raw_num = np.where(new_time_list <= down_raw)
            high_raw_num = np.where(high_new_time_list <= down_raw)[0]
        elif i == window_num - 1:
            up_raw = time_unique[int(per_scale * i)]
            raw_num = np.where(new_time_list >= up_raw)
            high_raw_num = np.where(high_new_time_list >= up_raw)[0]
        else:
            up_raw = time_unique[int(per_scale * i)]
            down_raw = time_unique[int(per_scale * (i + 1) - 1)]
            raw_num = np.where((new_time_list >= up_raw) & (new_time_list <= down_raw))
            high_raw_num = np.where((high_new_time_list >= up_raw) & (high_new_time_list <= down_raw))[0]
        edge_index_list = data[raw_num[0], 0:2]
        edges_index = [list(t) for t in set(tuple(_) for _ in edge_index_list)]
        adj = sp.coo_matrix((np.ones(len(edges_index)), ([x[0] for x in edges_index], [x[1] for x in edges_index])),
                                         shape=(node_number, node_number), dtype=np.float32)
        adj_list_single = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        matrix_mid = construct_adj(edges_index, node_number)
        lap_po, lap_ne = lap_pn(DAD_mid(matrix_mid), node_number)
        degree_single = torch.tensor(np.sum(matrix_mid, 1))
        high1 = choose_pos(high_raw_num, high_data)
        high1 = [list(t) for t in set(tuple(_) for _ in high1)]
        pos_single, neg_single = neg_sampling_random(ratio, seed_num, len(high1), high1, node_number)
        adj_list.append(adj_list_single)
        pos.append(pos_single)
        neg.append(neg_single)
        degree.append(degree_single)
        lap_p.append(lap_po)
        lap_n.append(lap_ne)
        clus.append(matrix_mid)
    return adj_list, pos, neg, per_scale, degree, lap_p, lap_n, clus


def choose_pos(high_raw_num, high_data):
    high1 = []
    for i in range(high_raw_num.shape[0]):
        z = high_data[high_raw_num[i]]
        high1.append(z[0, 0: high_data[high_raw_num[i]].shape[1] - 1])
    return high1


def choose_pos2(high_raw_num, high_data):
    high1 = []
    for i in range(len(high_raw_num)):
        z = high_data[high_raw_num[i]]
        high1.append(z)
    return high1


def preprocess_pos_neg(ratio, seed_num, node_number, window_num, high_data):
    high_new_time_list = [high_data[i][0, -1] for i in range(len(high_data))]
    time_unique = np.unique(high_new_time_list)
    per_scale = np.round(len(time_unique) / window_num)
    pos = []
    neg = []
    for i in range(int(window_num)):
        if i == 0:
            down_raw = time_unique[int(per_scale * (i + 1) - 1)]
            high_raw_num = np.where(high_new_time_list <= down_raw)[0]
        elif i == window_num - 1:
            up_raw = time_unique[int(per_scale * i)]
            high_raw_num = np.where(high_new_time_list >= up_raw)[0]
        else:
            up_raw = time_unique[int(per_scale * i)]
            down_raw = time_unique[int(per_scale * (i + 1) - 1)]
            high_raw_num = np.where((high_new_time_list >= up_raw) & (high_new_time_list <= down_raw))[0]
        high1 = [high_data[high_raw_num[i]][0, 0: high_data[high_raw_num[i]].shape[1] - 1] for i in range(high_raw_num.shape[0])]
        high1 = [list(t) for t in set(tuple(_) for _ in high1)]
        pos_single, neg_single = neg_sampling_random(ratio, seed_num, len(high1), high1, node_number)
        pos.append(pos_single)
        neg.append(neg_single)
    return pos, neg


def construct_adj(edges, node_num):
    adj_time = np.zeros(shape=(node_num, node_num))
    for i in range(len(edges)):
        adj_time[int(edges[i][0]), int(edges[i][1])] = 1
        adj_time[int(edges[i][1]), int(edges[i][0])] = 1
    return adj_time


def neg_sampling_random(ratio, seed_num, edge_num, true_sample, node_num):
    pos_sample = []
    neg_sample = []
    random.seed(seed_num)
    all_node = [i for i in range(node_num)]
    for i in range(edge_num):
        pos_ID = true_sample[i]
        pos_sample.append(list(map(int, pos_ID)))
        # half_num = math.ceil(len(pos_ID)/2)
        half_num = 1
        for j in range(ratio):
            neg = random.sample((set(all_node) - set(pos_ID)), int(len(pos_ID) - half_num))  # 1只保留一个
            neg_sample.append(list(map(int, [pos_ID[0]] + neg)))
            # neg_sample.append(np.array([pos_ID[0:int(half_num)] + neg]).astype(int).tolist())
    return pos_sample, neg_sample


def remove_edge(pos_samples):
    a = []
    for j in range(len(pos_samples)):
        pos_set = set(pos_samples[j])
        for k in range(len(pos_samples)):
            if j != k and set(pos_samples[k]) == pos_set and pos_set != []:
                a.append(k)
    new_pos = pos_samples.pop(a)
    return new_pos


# 得到正负拉普拉斯矩阵
def lap_pn(adj_matrix, node_number):
    adj_matrix_p = np.zeros((node_number, node_number))
    adj_matrix_n = np.zeros((node_number, node_number))
    for i in range(node_number):
        non_num = np.array(np.nonzero(adj_matrix[i, :]))
        mean_score = np.sum(adj_matrix[i, :]) / non_num.shape[1]
        big = np.where(adj_matrix[i, :] >= mean_score)
        adj_matrix_p[i, big] = adj_matrix[i, big]
        small = np.where(adj_matrix[i, :] < mean_score)
        adj_matrix_n[i, small] = adj_matrix[i, small]
    adj_matrix_p[adj_matrix_p != 0] = 1
    adj_matrix_n[adj_matrix_n != 0] = 1
    adj_matrix_p = DAD_mid(adj_matrix_p)
    adj_matrix_n = DAD_mid(adj_matrix_n)
    lap_matrix_p = np.eye(node_number) - adj_matrix_p
    lap_matrix_n = np.eye(node_number) - adj_matrix_n
    lap_matrix_p = torch.tensor(lap_matrix_p) #+ torch.eye(node_number)
    lap_matrix_n = torch.tensor(lap_matrix_n) #+ torch.eye(node_number)
    return lap_matrix_p, lap_matrix_n


def DAD_mid(matrix):
    matrix = np.array(matrix)
    D = np.diag(np.sum(matrix, axis=1) ** (-0.5))
    D[np.isinf(D)] = 0
    matrix_norm = np.dot(np.dot(D, matrix), D)
    return matrix_norm
