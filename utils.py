import networkx as nx
import numpy as np
from tqdm import tqdm

def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, max_len):
    seq = []
    for item_seq in tqdm(all_usr_pois):
        item_seq = np.asarray(item_seq)
        k = max_len - item_seq.shape[0]
        if k > 0:
            items = np.pad(item_seq, (0,k), 'constant', constant_values=0)
        else:
            items = item_seq[:max_len]
        seq.append(items)
    seq = np.asarray(seq)
    seq = np.reshape(seq, (-1, max_len))
    us_msks = seq.copy()
    us_msks[us_msks > 0] = 1
    return seq, us_msks

def data_masks_with_score(all_usr_pois, scores, max_len):
    seq = []
    scr = []
    for ziped in tqdm(zip(all_usr_pois, scores), total=len(all_usr_pois)):
        item_seq, score = ziped[0], ziped[1]
        item_seq = np.asarray(item_seq)
        score = np.asarray(score)
        k = max_len - item_seq.shape[0]
        if k > 0:
            items = np.pad(item_seq, (0,k), 'constant', constant_values=0)
            score_np = np.pad(score, (0,k), 'constant', constant_values=0)
        else:
            items = item_seq[:max_len]
            score_np = score[:max_len]
        seq.append(items)
        scr.append(score_np)
    seq = np.asarray(seq)
    seq = np.reshape(seq, (-1, max_len))
    scr = np.asarray(scr)
    scr = np.reshape(scr, (-1, max_len))
    us_msks = seq.copy()
    us_msks[us_msks > 0] = 1
    return seq, us_msks, scr

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, shuffle=False, graph=None, method=True):
        if len(data) == 4:
            inputs, mask, scores = data_masks_with_score(data[0], data[-1], data[2])
        else:
            inputs, mask = data_masks(data[0], data[2])
            scores = np.empty(1)
        
        self.inputs = inputs
        self.mask = mask
        self.targets = np.asarray(data[1])
        self.length = inputs.shape[0]
        self.shuffle = shuffle
        self.graph = graph
        self.scores = scores
        self.method = method

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            if self.scores.shape[0] != 1:
                self.scores = self.scores[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        scores = []
        if self.scores.shape[0] != 1:
            scores = self.scores[i]
        
        items, n_node, A, alias_inputs = [], [], [], []
        
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        
        max_n_node = np.max(n_node)
        if self.scores.shape[0] != 1:
            for _input in zip(inputs, scores):
                u_input, score = _input[0], _input[1]
                node = np.unique(u_input)
                items.append(node.tolist() + (max_n_node - len(node)) * [0])
                u_A = np.zeros((max_n_node, max_n_node))
                for i in np.arange(len(u_input) - 1):
                    if u_input[i + 1] == 0:
                        break
                    u = np.where(node == u_input[i])[0][0]
                    v = np.where(node == u_input[i + 1])[0][0]
                    k = (score[i + 1] - score[i])/2
                    if self.method:
                        if k >= 1:
                            u_A[u][v] = 2
                        elif k <= -1:
                            u_A[u][v] = -1
                        else:
                            u_A[u][v] = 1
                    else:
                        if score[i + 1] == score[i]:
                            u_A[u][v] = 1
                        elif score[i + 1] > score[i]:
                            u_A[u][v] = 2
                        else:
                            u_A[u][v] = 0.5
                    
                u_sum_in = np.sum(u_A, 0)
                u_sum_in[np.where(u_sum_in < 1)] = 1
                u_A_in = np.divide(u_A, u_sum_in)
                u_sum_out = np.sum(u_A, 1)
                u_sum_out[np.where(u_sum_out < 1)] = 1
                u_A_out = np.divide(u_A.transpose(), u_sum_out)
                u_A = np.concatenate([u_A_in, u_A_out]).transpose()
                A.append(u_A)
                alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        else:
            for u_input in inputs:
                node = np.unique(u_input)
                items.append(node.tolist() + (max_n_node - len(node)) * [0])
                u_A = np.zeros((max_n_node, max_n_node))
                for i in np.arange(len(u_input) - 1):
                    if u_input[i + 1] == 0:
                        break
                    u = np.where(node == u_input[i])[0][0]
                    v = np.where(node == u_input[i + 1])[0][0]
                    u_A[u][v] = 1
                u_sum_in = np.sum(u_A, 0)
                u_sum_in[np.where(u_sum_in == 0)] = 1
                u_A_in = np.divide(u_A, u_sum_in)
                u_sum_out = np.sum(u_A, 1)
                u_sum_out[np.where(u_sum_out == 0)] = 1
                u_A_out = np.divide(u_A.transpose(), u_sum_out)
                u_A = np.concatenate([u_A_in, u_A_out]).transpose()
                A.append(u_A)
                alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
