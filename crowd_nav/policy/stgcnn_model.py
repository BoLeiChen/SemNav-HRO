import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
from crowd_nav.policy.helpers import mlp
import torch.nn.functional as Func
import torch.distributions.multivariate_normal as torchdist
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
from crowd_nav.utils.memory import TrajectoryDataset, ReplayMemory

import torch.optim as optim

def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

class Social_StatePredictor(nn.Module):
    def __init__(self, graph_model, config, time_step, n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3, RGL_origin = False):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = True
        self.kinematics = config.action_space.kinematics
        self.RGL_origin = RGL_origin
        self.graph_model = graph_model
        self.human_motion_predictor_RGL = mlp(config.gcn.X_dim, config.model_predictive_rl.motion_predictor_dims)
        #就是要把这个换成更好的预测器
        #lsy:start
        #输入的维度是32，输出的维度是5
        self.human_motion_predictor = social_stgcnn(n_stgcnn,n_txpcnn,input_feat,output_feat,
                 seq_len,pred_seq_len,kernel_size)
        '''
        #social_stgcnn里面的参数需要定义一下这个文件里面都没有
        self.human_motion_predictor = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        output_feat=args.output_size,seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
        '''
        #lst:end
        self.time_step = time_step

    def forward(self, state, cur_memory, action, radius, detach=False):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        if action is None:
            # for training purpose
            next_robot_state = None
        else:
            next_robot_state = self.compute_next_state(state[0], action)
        if self.RGL_origin == False:
            dset_train = TrajectoryDataset(
                obs_len=3,
                pred_len=2,
                skip=1,norm_lap_matr=True)
            dset_train.create(cur_memory)
            #print('memory')
            #print(len(cur_memory.memory))
            loader_train = DataLoader(
                dset_train,
                batch_size=1, #This is irrelative to the args batch size parameter
                shuffle =True,
                num_workers=0)
            #print('loader')
            #print(len(loader_train))
            for cnt,batch in enumerate(loader_train):
                batch = [tensor.cuda() for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
                 loss_mask,V_obs,A_obs,V_tr,A_tr = batch
                num_of_objs = obs_traj_rel.shape[1]

                #Forward
                #V_obs = batch,seq,node,feat
                #V_obs_tmp = batch,feat,seq,node
                V_obs_tmp =V_obs.permute(0,3,1,2)

                V_pred,_ = self.human_motion_predictor(V_obs_tmp,A_obs.squeeze())
                # print(V_pred.shape)
                # torch.Size([1, 5, 12, 2])
                # torch.Size([12, 2, 5])
                V_pred = V_pred.permute(0,2,3,1)
                # torch.Size([1, 12, 2, 5])>>seq,node,feat
                # V_pred= torch.rand_like(V_tr).cuda()


                V_tr = V_tr.squeeze()
                A_tr = A_tr.squeeze()
                V_pred = V_pred.squeeze()
                num_of_objs = obs_traj_rel.shape[1]
                V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
                #print(V_pred.shape)

                #For now I have my bi-variate parameters
                #normx =  V_pred[:,:,0:1]
                #normy =  V_pred[:,:,1:2]
                sx = torch.exp(V_pred[:,:,2]) #sx
                sy = torch.exp(V_pred[:,:,3]) #sy
                corr = torch.tanh(V_pred[:,:,4]) #corr

                cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
                cov[:,:,0,0]= sx*sx
                cov[:,:,0,1]= corr*sx*sy
                cov[:,:,1,0]= corr*sx*sy
                cov[:,:,1,1]= sy*sy
                mean = V_pred[:,:,0:2]

                mvnormal = torchdist.MultivariateNormal(mean,cov)

                V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
                V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                         V_x[0,:,:].copy())

                V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
                V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                         V_x[-1,:,:].copy())

                V_pred = mvnormal.sample()
                #print(V_pred)

                #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
                V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                         V_x[-1,:,:].copy())

                pred = []
                for n in range(num_of_objs):

                    #new_V = V_pred_rel_to_abs[0,n:n+1,:].squeeze()
                    #new_V = np.append(V_pred_rel_to_abs[0,n:n+1,:].squeeze(),V_pred[1,n,0].data.cpu().numpy()/self.time_step)
                    #new_V = np.append(new_V,V_pred[1,n,1].data.cpu().numpy()/self.time_step)
                    new_V = np.append(V_pred_rel_to_abs[0,n:n+1,:].squeeze(),radius)
                    pred.append(new_V)
                    #pred[n].resize(pred[n].shape[0]+3)
                    #print(pred[n].shape)
                    #pred[n] = np.concatenate(pred[n],[V_pred[0,n,0]/self.time_step, V_pred[0,n,1]/self.time_step, radius])
                pred = np.array(pred)
                pred.resize(1,num_of_objs,3)
                #print('pre')
                #print(pred)
                pred = torch.tensor(pred)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                pred = pred.to(device)
                pred = pred.to(torch.float32)
                #这里需要的行人矩阵大小为1，行人，5（特征）
            #这里第三个维度应该还要包含速度，所以要计算一下更新next_human_states
        else:
            state_embedding = self.graph_model(state)
            pred = self.human_motion_predictor_RGL(state_embedding)[:, 1:, :]
        next_observation = [next_robot_state, pred]
        return next_observation

    def compute_next_state(self, robot_state, action):
        # currently it can not perform parallel computation
        if robot_state.shape[0] != 1:
            raise NotImplementedError

        # px, py, vx, vy, radius, gx, gy, v_pref, theta
        next_state = robot_state.clone().squeeze()
        if self.kinematics == 'holonomic':
            next_state[0] = next_state[0] + action.vx * self.time_step
            next_state[1] = next_state[1] + action.vy * self.time_step
            next_state[2] = action.vx
            next_state[3] = action.vy
        else:
            next_state[7] = next_state[7] + action.r
            next_state[0] = next_state[0] + np.cos(next_state[7]) * action.v * self.time_step
            next_state[1] = next_state[1] + np.sin(next_state[7]) * action.v * self.time_step
            next_state[2] = np.cos(next_state[7]) * action.v
            next_state[3] = np.sin(next_state[7]) * action.v

        return next_state.unsqueeze(0).unsqueeze(0)

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class social_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(social_stgcnn,self).__init__()
        self.trainable = True
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())


        
    def forward(self,v,a):

        for k in range(self.n_stgcnn):
            v,a = self.st_gcns[k](v,a)
            
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v
            
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        
        return v,a
