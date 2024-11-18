from time_process import *


def delete_one(relation, relation_time, relation_node):
    total_sample_num = relation.shape[0]
    k = 0
    set_0 = []
    for i in range(total_sample_num):
        k = k + relation[i]
        if relation[i] == 1:
            set_0.append(int(k - 1))
    sim_0 = np.where(relation == 1)[0]
    new_relation = np.delete(relation, sim_0, axis=0)
    new_time = np.delete(relation_time, sim_0, axis=0)
    mid_relation_node = np.delete(relation_node, set_0, axis=0)
    return new_relation, new_time, mid_relation_node


def resort_id(nodes, mid_relation_node):
    new_node = np.zeros_like(mid_relation_node)
    for i in range(nodes.shape[0]):
        hang = np.where(mid_relation_node == nodes[i])[0]
        new_node[hang] = i
    return new_node


def obtain_total(new_relation, new_relation_node, new_time):
    total_samples = []
    for i in range(new_relation.shape[0]):
        s = np.sum(new_relation[0: i])
        a = new_relation_node[int(s): int(s + new_relation[i])].reshape(1, int(new_relation[i]))
        b = new_time[i].reshape(1, 1)
        total_samples.append(np.concatenate((a[0, a.argsort()], b), axis=1))
    return total_samples


def split_train_vali_test(total_samples, orders, times):
    uni_times = np.unique(times)
    cut_time_train = uni_times[int(np.round((uni_times.shape[0] * 70)/100))]
    train_hang = np.where(times <= cut_time_train)[0]
    test_hang = np.where(times > cut_time_train)[0]
    second_train_hang = np.intersect1d(train_hang, np.where(orders == 2)[0])
    high_train_hang = np.setdiff1d(train_hang, np.where(orders == 2)[0])
    high_test_hang = np.setdiff1d(test_hang, np.where(orders == 2)[0])

    second_train_sample = [total_samples[second_train_hang[i]] for i in range(second_train_hang.shape[0])]
    high_train_sample = [total_samples[high_train_hang[i]] for i in range(high_train_hang.shape[0])]
    high_test_sample = [total_samples[high_test_hang[i]] for i in range(high_test_hang.shape[0])]
    return second_train_sample, high_train_sample,  high_test_sample, high_test_hang

