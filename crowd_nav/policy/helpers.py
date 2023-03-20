import torch
import torch.nn as nn


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        # 除了最后一层的其他层都要用ReLU函数激活一遍，最后一层要不要用ReLU函数激活一遍主要看last_relu这个参数
        if i != len(mlp_dims) - 2 or last_relu:
            #layers.append(nn.ReLU())
            i = i
    net = nn.Sequential(*layers)
    return net
