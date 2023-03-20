import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter
from crowd_nav.policy.helpers import mlp


class RGL(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        """ The current code might not be compatible with models trained with previous version
        """
        super().__init__()
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        final_state_dim = config.gcn.final_state_dim
        similarity_function = config.gcn.similarity_function
        layerwise_graph = config.gcn.layerwise_graph
        skip_connection = config.gcn.skip_connection

        # design choice

        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection

        logging.info('Similarity_func: {}'.format(self.similarity_function))
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)

        self.prelu = nn.PReLU()

        if self.similarity_function == 'embedded_gaussian':
            self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))
        elif self.similarity_function == 'concatenation':
            self.w_a = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)

        # TODO: try other dim size
        embedding_dim = self.X_dim
        self.Ws = torch.nn.ParameterList()
        #构造num_layer层的网络，网络输入大小是X_dim,输出大小是final_state_dim,隐层的大小都是embedding_dim
        #这个两层的网络用来对行人之间的关系进行建模
        for i in range(self.num_layer):
            if i == 0:
                self.Ws.append(Parameter(torch.randn(self.X_dim, embedding_dim)))
            elif i == self.num_layer - 1:
                self.Ws.append(Parameter(torch.randn(embedding_dim, final_state_dim)))
            else:
                self.Ws.append(Parameter(torch.randn(embedding_dim, embedding_dim)))

        # for visualization
        self.A = None

    #计算相似度函数
    def compute_similarity_matrix(self, X):
        if self.similarity_function == 'embedded_gaussian':
            A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'cosine':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'cosine_softmax':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = softmax(torch.div(A, norm_matrix), dim=2)
        elif self.similarity_function == 'concatenation':
            indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
            selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
            pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
            A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
            normalized_A = A
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError

        return normalized_A

    def forward(self, state):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """
        robot_state, human_states = state

        robot_state = torch.cat([robot_state[:, :, 0:5], robot_state[:, :, 7:9]], dim = 2)

        # compute feature matrix X，这个特征矩阵第一项是机器人，其他项是行人
        robot_state_embedings = self.w_r(robot_state)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)

        # compute matrix A
        if not self.layerwise_graph:
            normalized_A = self.compute_similarity_matrix(X)
            self.A = normalized_A[0, :, :].data.cpu().numpy()

        next_H = H = X
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.prelu.to(device)
        for i in range(self.num_layer):
            if self.layerwise_graph:
                A = self.compute_similarity_matrix(H)
                next_H = self.prelu(torch.matmul(torch.matmul(A, H), self.Ws[i]))
            else:
                next_H = self.prelu(torch.matmul(torch.matmul(normalized_A, H), self.Ws[i]))
                next_H = next_H
            if self.skip_connection:
                next_H += H
            H = next_H
        
        return next_H

class GAT_RL(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        """ The current code might not be compatible with models trained with previous version
        """
        super().__init__()
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        final_state_dim = config.gcn.final_state_dim
        similarity_function = config.gcn.similarity_function
        layerwise_graph = config.gcn.layerwise_graph
        skip_connection = config.gcn.skip_connection

        # design choice

        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gat0 = GraphAttentionLayer(self.X_dim, self.X_dim)
        self.gat1 = GraphAttentionLayer(self.X_dim, self.X_dim)

        logging.info('Similarity_func: {}'.format(self.similarity_function))
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)
        # for visualization
        self.attention_weights = None

    def compute_adjectory_matrix(self, state):
        robot_state = state[0]
        human_state = state[1]
        robot_num = robot_state.size()[1]
        human_num = human_state.size()[1]
        Num = robot_num + human_num
        adj = torch.ones((Num, Num))
        for i in range(robot_num, robot_num+human_num):
            adj[i][0] = 0
        adj = adj.repeat(robot_state.size()[0], 1, 1)
        return adj

    def forward(self, state):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """
        robot_state, human_states = state
        robot_state = torch.cat([robot_state[:, :, 0:5], robot_state[:, :, 7:9]], dim = 2)

        if human_states is None:
            robot_state_embedings = self.w_r(robot_state)
            adj = torch.ones((1, 1))
            adj = adj.repeat(robot_state.size()[0], 1, 1)
            X = robot_state_embedings
            if robot_state.shape[0]==1:
                H1, self.attention_weights = self.gat0(X, adj)
            else:
                H1, _ = self.gat0(X, adj)
            H2, _ = self.gat1(H1, adj)
            if self.skip_connection:
                output = H1 + H2 + X
            else:
                output = H2
            return output
        else:
            adj = self.compute_adjectory_matrix(state).to(self.device)
            # compute feature matrix X
            robot_state_embedings = self.w_r(robot_state)
            human_state_embedings = self.w_h(human_states)
            X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)
            if robot_state.shape[0]==1:
                H1, self.attention_weights = self.gat0(X, adj)
            else:
                H1, _ = self.gat0(X, adj)
            H2, _ = self.gat1(H1, adj)
            if self.skip_connection:
                output = H1 + H2 + X
            else:
                output = H2
            return output

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.w_a = mlp(2 * self.in_features, [2 * self.in_features, 1], last_relu=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.04)

    def forward(self, input, adj):

        # shape of input is batch_size, graph_size,feature_dims
        # shape of adj is batch_size, graph_size, graph_size
        assert len(input.shape) == 3
        assert len(adj.shape) == 3

        A = self.compute_similarity_matrix(input)
        e = self.leakyrelu(A)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        next_H = torch.matmul(attention, input)
        return next_H, attention[0, 0, :].data.cpu().numpy()

    def compute_similarity_matrix(self, X):
        indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
        selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1).to(self.device))
        pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
        A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
        return A

# class RGL(nn.Module):
#     def __init__(self, config, robot_state_dim, human_state_dim):
#         super().__init__()
#         self.multiagent_training = config.gcn.multiagent_training
#         num_layer = config.gcn.num_layer
#         X_dim = config.gcn.X_dim
#         wr_dims = config.gcn.wr_dims
#         wh_dims = config.gcn.wh_dims
#         final_state_dim = config.gcn.final_state_dim
#         gcn2_w1_dim = config.gcn.gcn2_w1_dim
#         similarity_function = config.gcn.similarity_function
#         layerwise_graph = config.gcn.layerwise_graph
#         skip_connection = config.gcn.skip_connection
#
#         # design choice
#
#         # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
#         self.similarity_function = similarity_function
#         logging.info('self.similarity_func: {}'.format(self.similarity_function))
#         self.robot_state_dim = robot_state_dim
#         self.human_state_dim = human_state_dim
#         self.num_layer = num_layer
#         self.X_dim = X_dim
#         self.layerwise_graph = layerwise_graph
#         self.skip_connection = skip_connection
#
#         self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
#         self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)
#
#         if self.similarity_function == 'embedded_gaussian':
#             self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))
#         elif self.similarity_function == 'concatenation':
#             self.w_a = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)
#
#         if num_layer == 1:
#             self.w1 = Parameter(torch.randn(self.X_dim, final_state_dim))
#         elif num_layer == 2:
#             self.w1 = Parameter(torch.randn(self.X_dim, gcn2_w1_dim))
#             self.w2 = Parameter(torch.randn(gcn2_w1_dim, final_state_dim))
#         else:
#             raise NotImplementedError
#
#         # for visualization
#         self.A = None
#
#     def compute_similarity_matrix(self, X):
#         if self.similarity_function == 'embedded_gaussian':
#             A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
#             normalized_A = softmax(A, dim=2)
#         elif self.similarity_function == 'gaussian':
#             A = torch.matmul(X, X.permute(0, 2, 1))
#             normalized_A = softmax(A, dim=2)
#         elif self.similarity_function == 'cosine':
#             A = torch.matmul(X, X.permute(0, 2, 1))
#             magnitudes = torch.norm(A, dim=2, keepdim=True)
#             norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
#             normalized_A = torch.div(A, norm_matrix)
#         elif self.similarity_function == 'cosine_softmax':
#             A = torch.matmul(X, X.permute(0, 2, 1))
#             magnitudes = torch.norm(A, dim=2, keepdim=True)
#             norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
#             normalized_A = softmax(torch.div(A, norm_matrix), dim=2)
#         elif self.similarity_function == 'concatenation':
#             indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
#             selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
#             pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
#             A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
#             normalized_A = A
#         elif self.similarity_function == 'squared':
#             A = torch.matmul(X, X.permute(0, 2, 1))
#             squared_A = A * A
#             normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
#         elif self.similarity_function == 'equal_attention':
#             normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
#         elif self.similarity_function == 'diagonal':
#             normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
#         else:
#             raise NotImplementedError
#
#         return normalized_A
#
#     def forward(self, state):
#         """
#         Embed current state tensor pair (robot_state, human_states) into a latent space
#         Each tensor is of shape (batch_size, # of agent, features)
#         :param state:
#         :return:
#         """
#         robot_state, human_states = state
#
#         # compute feature matrix X
#         robot_state_embedings = self.w_r(robot_state)
#         human_state_embedings = self.w_h(human_states)
#         X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)
#
#         # compute matrix A
#         normalized_A = self.compute_similarity_matrix(X)
#         self.A = normalized_A[0, :, :].data.cpu().numpy()
#
#         # graph convolution
#         if self.num_layer == 0:
#             state_embedding = X
#         elif self.num_layer == 1:
#             h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
#             state_embedding = h1
#         else:
#             # compute h1 and h2
#             if not self.skip_connection:
#                 h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
#             else:
#                 h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1)) + X
#             if self.layerwise_graph:
#                 normalized_A2 = self.compute_similarity_matrix(h1)
#             else:
#                 normalized_A2 = normalized_A
#             if not self.skip_connection:
#                 h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2))
#             else:
#                 h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2)) + h1
#             state_embedding = h2
#
#         return state_embedding
