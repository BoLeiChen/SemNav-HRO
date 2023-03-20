import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,\
            padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,\
            bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MapConv(nn.Module):

    def __init__(self, num_channel=23, conf=False, rc=False, use_local_map_semantic=True, layers=[2, 2, 2, 2], preconf=False):
        super(MapConv, self).__init__()

        self.conf = conf
        self.rc = rc
        self.local = use_local_map_semantic
        self.num_channel = num_channel

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        assert not preconf, "out-of-date preconf layers"
        if self.conf and preconf:
            self.preconf = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        else:
            self.preconf = None
        self.conv1 = nn.Conv2d(num_channel + int(self.conf) + int(self.rc), self.inplanes, kernel_size=7, stride=2, padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        self.l1 = nn.Linear(128 * 128, 1)
        self.l2 = nn.Linear(256 * 256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        obs = x[:, :self.num_channel,...]
        goal = x[:, self.num_channel:2*self.num_channel, ...]
        goal = self.pre_goal(goal)

        if self.conf:
            conf = x[:, 2*self.num_channel:, ...]
            conf = self.pre_conf(conf)
            x = torch.cat((obs, goal, conf), dim=1)
        else:
            x = torch.cat((obs, goal), dim=1)
'''
        if self.preconf is not None:
            conf = x[:, -1:, ...]
            normed_conf = self.preconf(conf)
            x = torch.cat((x[:, :-1, ...], normed_conf), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        x = self.conv4(x)

        q_map = x.cpu().detach().numpy().squeeze(0).squeeze(0)

        x = rearrange(x, 'b c h w -> b c (h w)')
        if self.local == False:
            q = self.l2(x).squeeze(2)
        else:
            q = self.l1(x).squeeze(2)

        return q, q_map

class Map_predictor(nn.Module):

    def __init__(self, num_channel, conf, rc, layers=[2, 2, 2, 2], preconf=False):
        super(Map_predictor, self).__init__()
        self.conf = conf
        self.rc = rc

        self.num_channel = num_channel

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        assert not preconf, "out-of-date preconf layers"
        if self.conf and preconf:
            self.preconf = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        else:
            self.preconf = None
        self.conv1 = nn.Conv2d(num_channel + int(self.conf) + int(self.rc), self.inplanes, kernel_size=7, stride=2,
                               padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 23, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.preconf is not None:
            conf = x[:, -2, ...]
            normed_conf = self.preconf(conf)
            x = torch.cat((x[:, :-1, ...], normed_conf), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)

        return x

