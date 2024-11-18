import numpy as np

from data_process import *


def load_data(ratio, seed_num, high_rate, window_num, dataset):
    print('Loading {} dataset...'.format(dataset))
    path = "new_data/" + dataset

    # loading original dataset
    relation = np.loadtxt(path + "/" + dataset + '-nverts.txt', delimiter=" ").astype(int)
    relation_node = np.loadtxt(path + "/" + dataset + '-simplices.txt', delimiter=" ").astype(int)
    relation_time = np.loadtxt(path + "/" + dataset + '-times.txt', delimiter=" ")

    # delete one-order node
    new_relation, mid_new_time, mid_relation_node = delete_one(relation, relation_time, relation_node)
    nodes = np.unique(mid_relation_node)
    node_number = nodes.shape[0]
    new_relation_node = resort_id(nodes, mid_relation_node)
    new_time = resort_id(np.unique(mid_new_time), mid_new_time)
    del relation, relation_node, relation_time, mid_new_time, mid_relation_node

    # all samples contain time
    total_samples = obtain_total(new_relation, new_relation_node, new_time)
    second_train_sample, high_train_sample, high_test_sample, high_test_hang = split_train_vali_test(total_samples, new_relation, new_time)
    del total_samples

    # training adj and three pair samples
    adj_train, pos_train, neg_train, per_scale0, degree, lap_p, lap_n, clus = preprocess_adj_pos_neg(ratio, seed_num, node_number, window_num, second_train_sample, high_train_sample)
    time_unique0 = np.unique(new_time[high_test_hang])
    test_window_num = np.round(len(time_unique0) / per_scale0)
    if test_window_num == 0:
        test_window_num = 1
    pos_test, neg_test = preprocess_pos_neg(ratio, seed_num, node_number, test_window_num, high_test_sample)
    del second_train_sample, high_train_sample, high_test_sample, high_test_hang

    # delete fully connected hyperedge
    adj_old = [adj_train[i].todense() for i in range(window_num)]
    final_pos_train, final_neg_train = [], []
    final_pos_test, final_neg_test = [], []
    for t in range(window_num):
       final_pos_tr, final_neg_tr = delete_circle(ratio, adj_old[t], pos_train[t], neg_train[t])
       final_pos_train.append(final_pos_tr)
       final_neg_train.append(final_neg_tr)
    for t in range(int(test_window_num)):
       final_pos_te, final_neg_te = delete_circle(ratio, adj_old[-1], pos_test[t], neg_test[t])
       final_pos_test.append(final_pos_te)
       final_neg_test.append(final_neg_te)
    del pos_train, neg_train, pos_test, neg_test

    adj = []
    for i in range(window_num):
        mid_adj = normalize(cir_edge(adj_train[i]))
        adj.append(sparse_mx_to_torch_sparse_tensor(mid_adj))

    high_adj = []
    line_graph = []
    inci = []
    for win in range(window_num):
        random.seed(seed_num)
        pos = final_pos_train[win]
        all_patterns = [i for i in range(len(pos))]
        samples_num = np.round(len(pos) * high_rate).astype(int)
        raw = random.sample(all_patterns, samples_num)
        high_adj_sample = [pos[raw[i]] for i in range(samples_num)]
        incidence_matrix_pre = np.zeros(shape=(node_number, samples_num))
        for k in range(samples_num):
            x = np.array(high_adj_sample[k]).astype(int)
            incidence_matrix_pre[x, k] = 1
        degree_edge_pre = np.sum(incidence_matrix_pre, axis=1)
        degree_matrix_pre = np.diag(degree_edge_pre)
        adj_matrix_pre = incidence_matrix_pre.dot(incidence_matrix_pre.T) - degree_matrix_pre
        A = DAD(cir_edge(adj_matrix_pre))
        high_adj.append(A)
        line_graph.append(sparse_mx_to_torch_sparse_tensor(normalize(cir_edge(hyper_adj(high_adj_sample)))))
        inci.append(sparse_mx_to_torch_sparse_tensor(normalize1(incidence(high_adj_sample, node_number))))

    return clus, degree, inci, adj, high_adj, final_pos_train, final_neg_train, final_pos_test, final_neg_test, node_number, test_window_num, lap_p, lap_n


def normalize1(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def incidence(sample, node_number):
    row = []
    colum = []
    value = []
    for i in range(len(sample)):
        for j in range(len(sample[i])):
            row.append(sample[i][j])
            colum.append(i)
            value.append(1)
    incidence_matrix = sp.csc_matrix((value, (row, colum)), shape=(node_number, len(sample)), dtype=np.float32)
    return incidence_matrix


def hyper_adj(samples):
    new_edge = []
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            interaction = set(samples[i]) & set(samples[j])
            if interaction:
                new_edge.append([i, j])
    adj = sp.coo_matrix((np.ones(len(new_edge)), ([x[0] for x in new_edge], [x[1] for x in new_edge])),
                        shape=(len(samples), len(samples)), dtype=np.float32)
    adj_list_single = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj_list_single


def generate_H(H):
    H = np.array(H.todense())
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE = np.mat(np.diag(np.power(DE, -1)))
    invDE[np.isinf(invDE)] = 0
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2[np.isinf(DV2)] = 0
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    G = DV2 * ((H * W * invDE * HT)) * DV2  #+ np.ones((H.shape[0], H.shape[0]))
    return G


def DAD(matrix):
    matrix = np.array(matrix)
    D = np.diag(np.sum(matrix, axis=1) ** (-0.5))
    D[np.isinf(D)] = 0
    matrix_norm = np.dot(np.dot(D, matrix), D)
    return torch.Tensor(matrix_norm)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


# 二阶关系的拉普拉斯矩阵
def cir_edge(data_matrix):
    adj_self_matrix = data_matrix + sp.eye(data_matrix.shape[0])
    return adj_self_matrix


def lap(matrix):
    degree_vector = np.sum(matrix, axis=1)
    degree_matrix = np.diag(degree_vector)
    lap_matrix = degree_matrix - matrix
    return torch.tensor(lap_matrix)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def delete_circle(ratio, adj, pos, neg):
    with_raw_train = []
    neg_with_raw_train = []
    for i in range(len(pos)):
        adj_ab = 0
        edge_number_diff = len(pos[i]) * (len(pos[i]) - 1) / 2
        for j in range(len(pos[i])):
            for k in range(j + 1, len(pos[i])):
                adj_ab = adj_ab + adj[int(pos[i][j]), int(pos[i][k])]
        if adj_ab != edge_number_diff:
            with_raw_train.append(i)
            for k1 in range(ratio):
                neg_with_raw_train.append(i * ratio + k1)
    final_pos = choose_pos2(with_raw_train, pos)
    final_neg = choose_pos2(neg_with_raw_train, neg)
    return final_pos, final_neg


