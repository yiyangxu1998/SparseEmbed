from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import AmazonDataset_local as dataset

from collections import Counter
import struct
import pickle
import math


# This is the network structure
class Net_1(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super(Net_1, self).__init__()

        self.feature_size = vocab_size
        self.embed_dim = embed_dim

        self.embed1 = nn.Linear(in_features=vocab_size, out_features=embed_dim)
        # self.embed2 = nn.Linear(in_features = embed_dim, out_features = embed_dim)
        self.bn_x = nn.BatchNorm1d(num_features=embed_dim)
        self.bn_y = nn.BatchNorm1d(num_features=embed_dim)
        self.cos = nn.CosineSimilarity()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = self.bn_x(self.embed1(x))
        y = self.bn_y(self.embed1(y))
        # x = self.embed(x)
        # y = self.embed(y)

        x = self.tanh(x)
        y = self.tanh(y)

        # r = self.cos(x, y)

        return x,y
        # return r
        # Let's have a softmax with

    def forward_query_embed(self, x):
        x = self.bn_x(self.embed1(x))
        x = self.tanh(x)

        return x

    def forward_asin_embed(self, y):
        y = self.bn_y(self.embed1(y))
        y = self.tanh(y)

        return y


class Net_2(nn.Module):
    def __init__(self, embed_dim_dense, embed_dim_sparse):
        super(Net_2, self).__init__()

        self.feature_size = embed_dim_dense
        self.embed_dim = embed_dim_sparse

        self.embed2 = nn.Linear(in_features=embed_dim_dense, out_features=embed_dim_sparse)
        # self.embed2 = nn.Linear(in_features = embed_dim, out_features = embed_dim)
        self.bn_x = nn.BatchNorm1d(num_features=embed_dim_sparse)
        self.bn_y = nn.BatchNorm1d(num_features=embed_dim_sparse)
        self.cos = nn.CosineSimilarity()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x, y):
    #     # evaluate the loss of sparse layer
    #     x = self.bn_x(self.embed2(x))
    #     y = self.bn_y(self.embed2(y))
    #     # x = self.embed(x)
    #     # y = self.embed(y)
    #
    #     x = self.tanh(x)
    #     y = self.tanh(y)
    #
    #     r_sparse = self.cos(x, y)
    #
    #     # evaluate the loss of the first max pooling
    #     x_pooling_1 = x
    #     y_pooling_1 = y
    #     pooling_num = math.ceil(self.embed_dim / 4)
    #     for i in range(pooling_num - 1):
    #         print(x.size())
    #         print(y.size())
    #         block_x = x_pooling_1[:, 4 * i:4 * i+4]
    #         block_y = y_pooling_1[:, 4 * i:4 * i+4]
    #         print("size")
    #         print(block_x.size)
    #         print("size")
    #         max_x, i_x = block_x.max(1)
    #         max_y, i_y = block_y.max(1)
    #         print(max_x.size())
    #         print(max_y.size())
    #         print("--")
    #         # max_x = max_x
    #         # max_y = max_y
    #         # print(max_x)
    #         # print("-------")
    #         # # print(torch.tensor(max_x.grad.clone(), requires_grad=False))
    #         # print("...")
    #         for row in range(64):
    #             max_xj = max_x[row]
    #             max_yj = max_y[row]
    #             for j in range(i * 4, i * 4 + 4):
    #                 # print(x_pooling_1[j].detach())
    #                 # print(max_x)
    #                 # print("")
    #                 # if x_pooling_1[j] != max_x:
    #                 #     x_pooling_1[j] = 0
    #                 # if y_pooling_1[j] != max_y:
    #                 #     y_pooling_1[j] = 0
    #                 x_help = x_pooling_1[row][j]
    #                 y_help = y_pooling_1[row][j]
    #                 # max_xj = max_x[row]
    #                 # max_yj = max_y[row]
    #                 # if not torch.eq(x_help, max_x):
    #                 # print(x_help)
    #                 # print(max_xj)
    #                 # print(torch.eq(x_help, max_xj))
    #                 print("--")
    #                 print(x_pooling_1[row][j])
    #                 if not torch.eq(x_help, max_xj):
    #                 # if not x_help is max_x[j]:
    #                     x_pooling_1[row][j] = 0
    #                 else:
    #                     print("true")
    #                 print(x_pooling_1[row][j])
    #                 print("--")
    #                 if not torch.eq(y_help, max_yj):
    #                     y_pooling_1[row][j] = 0
    #     if not 4 * (pooling_num - 1) > self.embed_dim:
    #         block_x = x_pooling_1[:, 4 * (pooling_num - 1):self.embed_dim]
    #         block_y = y_pooling_1[:, 4 * (pooling_num - 1):self.embed_dim]
    #         max_x, i_x = block_x.max(1)
    #         max_y, i_y = block_y.max(1)
    #         # max_x = max_x
    #         # max_y = max_y
    #         for row in range(64):
    #             max_xj = max_x[row]
    #             max_yj = max_y[row]
    #             for j in range(4 * (pooling_num - 1), pooling_num):
    #                 x_help = x_pooling_1[row][j]
    #                 y_help = y_pooling_1[row][j]
    #                 # max_xj = max_x[j]
    #                 # max_yj = max_y[j]
    #                 # if x_pooling_1[j] != max_x:
    #                 #     x_pooling_1[j] = 0
    #                 # if y_pooling_1[j] != max_y:
    #                 #     y_pooling_1[j] = 0
    #                 if not torch.eq(x_help, max_xj):
    #                     x_pooling_1[row][j] = 0
    #                 if not torch.eq(y_help, max_yj):
    #                     y_pooling_1[row][j] = 0
    #
    #     r_pooling_1 = self.cos(x_pooling_1, y_pooling_1)
    #
    #     # # evaluate the loss of the second max pooling
    #     # x_pooling_2 =
    #     # y_pooling_2 =
    #     #
    #     # # weight the losses
    #     # r = 1.0/3.0 * r_sparse +
    #
    #     return r_pooling_1
    #     # Let's have a softmax with

    def forward(self, x, y):
        # evaluate the loss of sparse layer
        x = self.bn_x(self.embed2(x))
        y = self.bn_y(self.embed2(y))
        # x = self.embed(x)
        # y = self.embed(y)

        x = self.tanh(x)
        y = self.tanh(y)

        r_sparse = self.cos(x, y)

        # evaluate the loss of the first max pooling
        # x_pooling_1 = x
        # y_pooling_1 = y
        pooling_num = math.ceil(self.embed_dim / 4)
        for i in range(pooling_num - 1):
            # print(x.size())
            # print(y.size())
            block_x = x[:, 4 * i:4 * i+4]
            block_y = y[:, 4 * i:4 * i+4]
            # print("size")
            # print(block_x.size)
            # print("size")
            max_x, i_x = block_x.max(1)
            max_y, i_y = block_y.max(1)
            # print(max_x.size())
            # print(max_y.size())
            # print("--")
            # max_x = max_x
            # max_y = max_y
            # print(max_x)
            # print("-------")
            # # print(torch.tensor(max_x.grad.clone(), requires_grad=False))
            # print("...")
            for row in range(64):
                max_xj = max_x[row]
                max_yj = max_y[row]
                for j in range(i * 4, i * 4 + 4):
                    # print(x_pooling_1[j].detach())
                    # print(max_x)
                    # print("")
                    # if x_pooling_1[j] != max_x:
                    #     x_pooling_1[j] = 0
                    # if y_pooling_1[j] != max_y:
                    #     y_pooling_1[j] = 0
                    x_help = x[row][j]
                    y_help = y[row][j]
                    # max_xj = max_x[row]
                    # max_yj = max_y[row]
                    # if not torch.eq(x_help, max_x):
                    # print(x_help)
                    # print(max_xj)
                    # print(torch.eq(x_help, max_xj))
                    # print("--")
                    # print(x[row][j])
                    if not torch.eq(x_help, max_xj):
                    # if not x_help is max_x[j]:
                        x[row][j] = 0
                    # else:
                    #     print("true")
                    # print(x[row][j])
                    # print("--")
                    if not torch.eq(y_help, max_yj):
                        y[row][j] = 0
        if not 4 * (pooling_num - 1) > self.embed_dim:
            block_x = x[:, 4 * (pooling_num - 1):self.embed_dim]
            block_y = y[:, 4 * (pooling_num - 1):self.embed_dim]
            max_x, i_x = block_x.max(1)
            max_y, i_y = block_y.max(1)
            # max_x = max_x
            # max_y = max_y
            for row in range(64):
                max_xj = max_x[row]
                max_yj = max_y[row]
                for j in range(4 * (pooling_num - 1), self.embed_dim):
                    x_help = x[row][j]
                    y_help = y[row][j]
                    # max_xj = max_x[j]
                    # max_yj = max_y[j]
                    # if x_pooling_1[j] != max_x:
                    #     x_pooling_1[j] = 0
                    # if y_pooling_1[j] != max_y:
                    #     y_pooling_1[j] = 0
                    if not torch.eq(x_help, max_xj):
                        x[row][j] = 0
                    if not torch.eq(y_help, max_yj):
                        y[row][j] = 0

        r_pooling_1 = self.cos(x, y)

        # # evaluate the loss of the second max pooling
        # x_pooling_2 =
        # y_pooling_2 =
        #
        # # weight the losses
        # r = 1.0/3.0 * r_sparse +

        return r_pooling_1
        # Let's have a softmax with

    def forward_query_embed(self, x):
        x = self.bn_x(self.embed1(x))
        x = self.tanh(x)

        return x

    def forward_asin_embed(self, y):
        y = self.bn_y(self.embed1(y))
        y = self.tanh(y)

        return y


def extend_hinge_loss(output, target):
    return - torch.sum(torch.mul(output, target))


def precision_at_k(ground_truth_batch, predictions_batch, reduction='mean'):
    """

    :param ground_truth_batch:
    :param predictions_batch:
    :param reduction:
    :return:
    """
    k_max = 5
    _, indices = torch.sort(predictions_batch, descending=True)
    top_k = indices[:, :k_max]
    top_k_in_ground_truth = torch.gather(ground_truth_batch, 1, top_k)
    n = ground_truth_batch.shape[0]
    if reduction == 'mean':
        precision_5 = top_k_in_ground_truth[:, :5].gt(0).sum().item() / (5 * n)
        precision_3 = top_k_in_ground_truth[:, :3].gt(0).sum().item() / (3 * n)
        precision_1 = top_k_in_ground_truth[:, :1].gt(0).sum().item() / (1 * n)
    elif reduction == 'sum':
        precision_5 = top_k_in_ground_truth[:, :5].gt(0).sum().item() / (5)
        precision_3 = top_k_in_ground_truth[:, :3].gt(0).sum().item() / (3)
        precision_1 = top_k_in_ground_truth[:, :1].gt(0).sum().item() / (1)
    else:
        precision_5 = top_k_in_ground_truth[:, :5].gt(0).sum(1) / 5
        precision_3 = top_k_in_ground_truth[:, :3].gt(0).sum(1) / 3
        precision_1 = top_k_in_ground_truth[:, :1].gt(0).sum(1) / 1
    return precision_1, precision_3, precision_5


def train_1(args, model, device, train_loader, optimizer, epoch):
    model.train()
    print("start to train Net_1")

    for batch_idx, batch in enumerate(train_loader):

        data_q = torch.sparse.FloatTensor(torch.LongTensor(batch['query_feature_index_batch']),
                                          torch.Tensor(batch['query_feature_value_batch']),
                                          torch.Size([args.batch_size, model.feature_size])).to(device).to_dense()
        data_t = torch.sparse.FloatTensor(torch.LongTensor(batch['title_feature_index_batch']),
                                          torch.Tensor(batch['title_feature_value_batch']),
                                          torch.Size([args.batch_size, model.feature_size])).to(device).to_dense()
        '''
        print(batch['query_feature_index_batch'])
        print(batch['title_feature_index_batch'])
        data_q = torch.LongTensor(batch['query_feature_index_batch']).to(device)
        data_t = torch.LongTensor(batch['title_feature_index_batch']).to(device)
        '''
        target = torch.FloatTensor(batch['label']).to(device)
        # target[target == -1] = 0
        # target = target.to(device)

        optimizer.zero_grad()
        output_q, output_t = model(data_q, data_t)
        # output_cos = model.cos(output_q, output_t)
        cos = nn.CosineSimilarity(dim=1)
        output_cos = cos(output_q, output_t)

        # print (output)
        # print (target)
        loss = extend_hinge_loss(output_cos, target)
        # loss = F.binary_cross_entropy(output, target)
        # loss = F.hinge_embedding_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_2(Net_1_output_q, Net_1_output_t, Net_1_target, args, model, device, train_loader, optimizer, epoch):
    model.train()
    print("start to train Net_2")

    batch_size = args.batch_size
    size = list(Net_1_output_q.size())[0]  # get the number of data points
    shuffle_index = torch.randperm(size)
    shuffle_q = Net_1_output_q[shuffle_index, :]
    shuffle_t = Net_1_output_t[shuffle_index, :]
    shuffle_target = Net_1_target[shuffle_index]
    batch_num = math.ceil(size / batch_size)
    for i in range(batch_num - 1):
        batch_q = shuffle_q[i * batch_size: (i + 1) * batch_size, :]
        batch_t = shuffle_t[i * batch_size: (i + 1) * batch_size, :]
        target = shuffle_target[i * batch_size: (i + 1) * batch_size]
        optimizer.zero_grad()
        output = model(batch_q, batch_t)

        loss = extend_hinge_loss(output, target)
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))
    batch_q = shuffle_q[batch_size * (batch_num - 1): size, :]
    batch_t = shuffle_t[batch_size * (batch_num - 1): size, :]
    target = shuffle_target[batch_size * (batch_num - 1): size]
    optimizer.zero_grad()
    output = model(batch_q, batch_t)

    loss = extend_hinge_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_num % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_num * batch_size, len(train_loader.dataset),
                   100. * batch_num / len(train_loader), loss.item()))


def test_1(args, model, device, size_topRanking=20):
    model.eval()
    correct = 0
    test_loss = 0
    query_file = args.query_file
    q = pickle.load(open(query_file, 'rb'))
    queries = []
    for i in q:
        queries.append(i)

    title_file = args.title_file
    t = pickle.load(open(title_file, 'rb'))
    titles = []
    for i in t:
        titles.append(i)

    print("Start to construct query-title pair frequency ranking")
    knn_file = args.test_train_knn_file
    test_train_knn = open(knn_file, 'rb')
    knn = pickle.load(test_train_knn)  # dictionary
    # print(knn)
    # print(type(knn))
    # knn_content = test_train_knn.read()
    # print(struct.unpack("i" * ((len(knn_content) - 24) // 4), knn_content[20:-4]))
    # print(knn_content)
    frequency = {}
    frequency_topRanking = {}
    for query in knn:
        frequency[query] = {}
        # frequency[query]["count"] = 0
        # frequency[query]["smallest"] = 0
        titles = knn[query]
        for title in titles:
            if title not in frequency[query]:
                frequency[query][title] = 0
            frequency[query][title] = frequency[query][title] + 1

        k = Counter(frequency[query])
        frequency_topRanking[query] = k.most_common(size_topRanking)  # list of most frequent query-title pairs  [[],[],[],...]

    # then use the model to compute the 20 most similar titles for each query (cosine similarity)
    all_queries_vectors = pickle.load(open(args.test_train_qlist_file, 'rb'))
    all_titles_vectors = pickle.load(open(args.test_train_alist_file, 'rb'))

    matching_test = {}
    matching_ranking_test = {}

    print("Start to test training model")
    with torch.no_grad():
        ccccc = 1
        for q in queries:
            matching_test[q] = {}
            query_word_embedding = all_queries_vectors[q]
            # print(query_word_embedding)
            q_feature_value = []
            q_feature_index = [[],[]]
            q_feature_index[0].extend([0] * len(query_word_embedding))
            # q_feature_index[1].extend(list(range(model.feature_size)))
            # q_feature_index = list(range(model.feature_size))
            for i in query_word_embedding:
                q_feature_index[1].append(i)
                q_feature_value.append(query_word_embedding[i])
            q_tensor = torch.sparse.FloatTensor(torch.LongTensor(q_feature_index),
                                               torch.Tensor(q_feature_value),
                                               torch.Size([1,model.feature_size])).to(device)
            for t in titles:
                title_word_embedding = all_titles_vectors[t]
                t_feature_value = []
                t_feature_index = [[], []]
                t_feature_index[0].extend([0] * len(title_word_embedding))
                # t_feature_index[1].extend(list(range(model.feature_size)))
                # t_feature_index = list(range(model.feature_size))
                for i in title_word_embedding:
                    t_feature_index[1].append(i)
                    t_feature_value.append(title_word_embedding[i])
                t_tensor = torch.sparse.FloatTensor(torch.LongTensor(t_feature_index),
                                                    torch.Tensor(t_feature_value),
                                                    torch.Size([1, model.feature_size])).to(device)
                output_q, output_t = model(q_tensor,t_tensor)
                output_cos = model.cos(output_q, output_t)
                # if ccccc < 10:
                    # print(output_q)
                    # print(output_t)
                    # print(output_cos)
                matching_test[q][t] = output_cos
                ccccc = ccccc + 1

            k = Counter(matching_test[q])
            temp = k.most_common(size_topRanking)
            matching_ranking_test[q] = []
            for pair in temp:
                matching_ranking_test[q].append(pair[0])

    # compare testing result through the model with true frequencies
    num_true = 0
    # num_we_say_true = 0
    num_we_say_true_are_true = 0
    for query in frequency_topRanking:
        if query != "- 20th Century Masters: Millennium Collection":
            # print(query)
            # print(frequency_topRanking[query])
            # print("")
            num_true = num_true + len(frequency_topRanking[query])
            for title_frequency_pair in frequency_topRanking[query]:
                # print(title_frequency_pair)
                if title_frequency_pair[0] in matching_ranking_test[query]:
                    num_we_say_true_are_true += 1
            print(num_we_say_true_are_true * 1.0 / (num_true * 1.0))
    recall = num_we_say_true_are_true * 1.0 / (num_true * 1.0)
    print("There're %d true cases, among which %d are given true by the model. The recall is %d" % (num_true, num_we_say_true_are_true, recall))
    return


def test_2(args, model, device, size_topRanking=20):
    model.eval()
    correct = 0
    test_loss = 0
    query_file = args.query_file
    q = pickle.load(open(query_file, 'rb'))
    queries = []
    for i in q:
        queries.append(i)

    title_file = args.title_file
    t = pickle.load(open(title_file, 'rb'))
    titles = []
    for i in t:
        titles.append(i)

    print("Start to construct query-title pair frequency ranking")
    knn_file = args.test_train_knn_file
    test_train_knn = open(knn_file, 'rb')
    knn = pickle.load(test_train_knn)  # dictionary
    # print(knn)
    # print(type(knn))
    # knn_content = test_train_knn.read()
    # print(struct.unpack("i" * ((len(knn_content) - 24) // 4), knn_content[20:-4]))
    # print(knn_content)
    frequency = {}
    frequency_topRanking = {}
    for query in knn:
        frequency[query] = {}
        # frequency[query]["count"] = 0
        # frequency[query]["smallest"] = 0
        titles = knn[query]
        for title in titles:
            if title not in frequency[query]:
                frequency[query][title] = 0
            frequency[query][title] = frequency[query][title] + 1

        k = Counter(frequency[query])
        frequency_topRanking[query] = k.most_common(size_topRanking)  # list of most frequent query-title pairs  [[],[],[],...]

    # then use the model to compute the 20 most similar titles for each query (cosine similarity)
    all_queries_vectors = pickle.load(open(args.test_train_qlist_file, 'rb'))
    all_titles_vectors = pickle.load(open(args.test_train_alist_file, 'rb'))

    matching_test = {}
    matching_ranking_test = {}

    print("Start to test training model")
    with torch.no_grad():
        for q in queries:
            matching_test[q] = {}
            query_word_embedding = all_queries_vectors[q]
            # print(query_word_embedding)
            q_feature_value = []
            q_feature_index = [[],[]]
            q_feature_index[0].extend([0] * len(query_word_embedding))
            # q_feature_index[1].extend(list(range(model.feature_size)))
            # q_feature_index = list(range(model.feature_size))
            for i in query_word_embedding:
                q_feature_index[1].append(i)
                q_feature_value.append(query_word_embedding[i])
            q_tensor = torch.sparse.FloatTensor(torch.LongTensor(q_feature_index),
                                               torch.Tensor(q_feature_value),
                                               torch.Size([1,model.feature_size])).to(device)
            for t in titles:
                title_word_embedding = all_titles_vectors[t]
                t_feature_value = []
                t_feature_index = [[], []]
                t_feature_index[0].extend([0] * len(title_word_embedding))
                # t_feature_index[1].extend(list(range(model.feature_size)))
                # t_feature_index = list(range(model.feature_size))
                for i in title_word_embedding:
                    t_feature_index[1].append(i)
                    t_feature_value.append(title_word_embedding[i])
                t_tensor = torch.sparse.FloatTensor(torch.LongTensor(t_feature_index),
                                                    torch.Tensor(t_feature_value),
                                                    torch.Size([1, model.feature_size])).to(device)
                output = model(q_tensor,t_tensor)
                matching_test[q][t] = output

            k = Counter(matching_test[q])
            temp = k.most_common(size_topRanking)
            matching_ranking_test[q] = []
            for pair in temp:
                matching_ranking_test[q].append(pair[0])

    # compare testing result through the model with true frequencies
    num_true = 0
    # num_we_say_true = 0
    num_we_say_true_are_true = 0
    for query in frequency_topRanking:
        if query != "- 20th Century Masters: Millennium Collection":
            # print(query)
            # print(frequency_topRanking[query])
            # print("")
            num_true = num_true + len(frequency_topRanking[query])
            for title_frequency_pair in frequency_topRanking[query]:
                # print(title_frequency_pair)
                if title_frequency_pair[0] in matching_ranking_test[query]:
                    num_we_say_true_are_true += 1
            print(num_we_say_true_are_true * 1.0 / (num_true * 1.0))
    recall = num_we_say_true_are_true * 1.0 / (num_true * 1.0)
    print("There're %d true cases, among which %d are given true by the model. The recall is %d" % (num_true, num_we_say_true_are_true, recall))
    return


def get_Net_1_output(args, model, train_loader_1):
    """
    Gather all batches and return the full result of the Net_1
    :param args:
    :param model:
    :param train_loader_1:
    :return:
    """
    model_1.eval()
    with torch.no_grad():
        all_batches_q = torch.tensor([args.batch_size, args.embed_dim_dense], requires_grad=False)  # a 2-D tensor
        all_batches_t = torch.tensor([args.batch_size, args.embed_dim_dense], requires_grad=False)  # a 2-D tensor
        all_batches_target = torch.tensor([args.batch_size, 1])
        for batch_idx, batch in enumerate(train_loader_1):
            data_q = torch.sparse.FloatTensor(torch.LongTensor(batch['query_feature_index_batch']),
                                              torch.Tensor(batch['query_feature_value_batch']),
                                              torch.Size([args.batch_size, model.feature_size])).to(device).to_dense()
            data_t = torch.sparse.FloatTensor(torch.LongTensor(batch['title_feature_index_batch']),
                                              torch.Tensor(batch['title_feature_value_batch']),
                                              torch.Size([args.batch_size, model.feature_size])).to(device).to_dense()
            target = torch.FloatTensor(batch['label']).to(device)

            output_q, output_t = model(data_q, data_t)  # output of a whole batch
            if batch_idx == 0:
                all_batches_q = output_q
                all_batches_t = output_t
                all_batches_target = target
            else:
                # print(all_batches_q.size())
                # print(output_q.size())
                all_batches_q = torch.cat([all_batches_q, output_q], dim=0)
                all_batches_t = torch.cat([all_batches_t, output_t], dim=0)
                all_batches_target = torch.cat([all_batches_target, target], dim=0)
    all_batches_q = all_batches_q.type(torch.FloatTensor)
    all_batches_t = all_batches_t.type(torch.FloatTensor)
    all_batches_target = all_batches_target.type(torch.FloatTensor)
    return all_batches_q, all_batches_t, all_batches_target







# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         i = 0
#         for data in test_loader:
#             # query_feature_value = data["query_feature_value"]
#             # query_feature_index = data["query_feature_index"]
#             # title_feature_value = data["title_feature_value"]
#             # title_feature_index = data["title_feature_index"]
#             # target = data["label"]
#             #
#             # two_D_index_q = [[],[]]
#             # two_D_index_q[0].extend([0]* len(query_feature_index))
#             # two_D_index_q[1].extend(query_feature_index)
#             # input_q = torch.sparse.FloatTensor(torch.LongTensor(two_D_index_q),
#             #                                    torch.Tensor(query_feature_value),
#             #                                    torch.Size(1,model.feature_size)).to(device)
#             # two_D_index_t = [[], []]
#             # two_D_index_t[0].extend([0] * len(title_feature_index))
#             # two_D_index_t[1].extend(title_feature_index)
#             # input_t = torch.sparse.FloatTensor(torch.LongTensor(two_D_index_t),
#             #                                    torch.Tensor(title_feature_value),
#             #                                    torch.Size(1, model.feature_size)).to(device)
#             # output = model(input_q, input_t)
#             # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             # correct += pred.eq(target.view_as(pred)).sum().item()
#
#             query_feature_value_batch = data["query_feature_value_batch"]
#             query_feature_index_batch = data["query_feature_index_batch"]
#             title_feature_value_batch = data["title_feature_value_batch"]
#             title_feature_index_batch = data["title_feature_index_batch"]
#             target = data["label"]
#
#             input_q = torch.sparse.FloatTensor(torch.LongTensor(query_feature_index_batch),
#                                                torch.Tensor(query_feature_value_batch),
#                                                torch.Size([args.batch_size, model.feature_size])).to(device).to_dense()
#             input_t = torch.sparse.FloatTensor(torch.LongTensor(title_feature_index_batch),
#                                                torch.Tensor(title_feature_value_batch),
#                                                torch.Size([args.batch_size, model.feature_size])).to(device).to_dense()
#             output = model(input_q, input_t)
#             print(output.size())
#             print(output)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#             # compute precisions
#             test_p1, test_p3, test_p5 = precision_at_k(target, output, 'mean')
#
#             if i < 10:
#                 print(test_loss)
#                 print(pred)
#                 print(output)
#                 print(correct)
#             i += 1
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs-1', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10) for Net_1')
parser.add_argument('--epochs-2', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10) for Net_2')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train-file', type=str, metavar='N',
                    help='The Path of the training data')
parser.add_argument('--test-file', type=str, metavar='N',
                    help='The Path of the testing data')
parser.add_argument('--vocab-size', type=int, default=23486, metavar='N',
                    help='The Vocab size of the data')
parser.add_argument('--embed-dim-dense', type=int, default=256, metavar='N',
                    help='The dimension of the Embedding vector')
parser.add_argument('--embed-dim-sparse', type=int, default=10240, metavar='N',
                    help='The dimension of the second Embedding vector')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

# parser.add_argument('--matching-frequency-file', type=str, metavar='N',
#                     help='The path of file containg the matching frequencies between querys and titles')
parser.add_argument('--query-file', type=str, metavar='N',
                    help='The path of all queries')
parser.add_argument('--title-file', type=str, metavar='N',
                    help='The path of all queries')
# parser.add_argument('--knn-file', type=str, metavar='N',
                    # help='The query-title matchings stored in a dictionary, including both testing and training data')
parser.add_argument('--test-train-knn-file', type=str, metavar='N',
                    help='The complete query-title matching including both training and testing files')
parser.add_argument('--test-train-qlist-file', type=str, metavar='N',
                    help='Matching of query to vectors representations')
parser.add_argument('--test-train-alist-file', type=str, metavar='N',
                    help='Matching of title to vectors representations')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader_1 = torch.utils.data.DataLoader(
    dataset.train_dataset(args.train_file),
    batch_size=args.batch_size, shuffle=True, collate_fn=dataset.AmazonDataset_collate, drop_last=True)
test_loader_1 = torch.utils.data.DataLoader(
    dataset.test_dataset(args.test_file),
    batch_size=args.test_batch_size, shuffle=True, collate_fn=dataset.AmazonDataset_collate, drop_last=True)

model_1 = Net_1(embed_dim=args.embed_dim_dense, vocab_size=args.vocab_size).to(device)
optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

for epoch in range(1, args.epochs_1 + 1):
    train_1(args, model_1, device, train_loader_1, optimizer, epoch)
    # test(args, model, device, test_loader)
    test_1(args, model_1, device)


# prepare the input data for the second Net training
model_2 = Net_2(embed_dim_dense=args.embed_dim_dense, embed_dim_sparse=args.embed_dim_sparse)
Net_1_q, Net_1_t, Net_1_target = get_Net_1_output(args, model_1, train_loader_1)
optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

for epoch in range(1, args.epochs_2 + 1):
    train_2(Net_1_q, Net_1_t, Net_1_target, args, model_2, device, train_loader_1, optimizer, epoch)
    test_2(args, model_2, device)


#
# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                         help='SGD momentum (default: 0.5)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     parser.add_argument('--train-file', type=str, metavar='N',
#                         help='The Path of the training data')
#     parser.add_argument('--test-file', type=str, metavar='N',
#                         help='The Path of the testing data')
#     parser.add_argument('--vocab-size', type=int, default=23486, metavar='N',
#                         help='The Vocab size of the data')
#     parser.add_argument('--embed-dim', type=int, default=256, metavar='N',
#                         help='The dimension of the Embedding vector')
#
#     parser.add_argument('--save-model', action='store_true', default=False,
#                         help='For Saving the current Model')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#
#     torch.manual_seed(args.seed)
#
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         dataset.train_dataset(args.train_file),
#         batch_size=args.batch_size, shuffle=True, collate_fn=dataset.AmazonDataset_collate, drop_last=True)
#     test_loader = torch.utils.data.DataLoader(
#         dataset.test_dataset(args.test_file),
#         batch_size=args.test_batch_size, shuffle=True, collate_fn=dataset.AmazonDataset_collate, drop_last=True)
#
#     model = Net(embed_dim=args.embed_dim, vocab_size=args.vocab_size).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         # test(args, model, device, test_loader)
#
#     if (args.save_model):
#         torch.save(model.state_dict(), "embed.pt")
#
#
# if __name__ == '__main__':
#     main()
