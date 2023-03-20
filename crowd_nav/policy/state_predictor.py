import torch
import torch.nn as nn
import numpy as np
from crowd_nav.policy.helpers import mlp
from crowd_nav.policy.mapconv import Map_predictor
import torch.nn.functional as F
from einops import asnumpy, repeat
from PIL import Image, ImageFont, ImageDraw
from crowd_nav.poni.constants import d3_40_colors_rgb
import matplotlib.pyplot as plt

class StatePredictor(nn.Module):
    def __init__(self, config, graph_model, time_step):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = True
        self.kinematics = config.action_space.kinematics
        self.graph_model = graph_model
        #就是要把这个换成更好的预测器
        #lsy:start
        #输入的维度是32，输出的维度是5
        self.human_motion_predictor = mlp(config.gcn.X_dim, config.model_predictive_rl.motion_predictor_dims)
        '''
        #social_stgcnn里面的参数需要定义一下这个文件里面都没有
        self.human_motion_predictor = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        output_feat=args.output_size,seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
        '''
        #lst:end
        self.time_step = time_step

    def forward(self, state, action, detach=False):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        state_embedding = self.graph_model(state)[:, 0, :]

        if detach:
            state_embedding = state_embedding.detach()
        if action is None:
            # for training purpose
            next_robot_state = None
        else:
            next_robot_state = self.compute_next_state(state[0], action)
        next_human_states = self.human_motion_predictor(state_embedding)[:, 1:, :]

        next_observation = [next_robot_state, next_human_states]
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

class MapPredictor(nn.Module):
    def __init__(self, time_step, use_local_map_semantic):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = True
        self.time_step = time_step
        self.mappredictor = Map_predictor(num_channel=23, conf=False, rc=False)
        self.local = use_local_map_semantic

    def forward(self, local_map, action):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """
        if self.local:
            mappredictor_input = local_map[:, 0:23:, :, :]
            #mappredictor_input = self.create_mappredictor_input(local_map[:,0:23:,:,:], action)  # local_map (b, 23, 128, 128)  action (b, 1, 2)
        else:
            mappredictor_input = local_map[:,0:23:,:,:]

        next_local_map = self.mappredictor(mappredictor_input)
        next_local_map = torch.cat([next_local_map, local_map[:,23:,:,:]], dim = 1)

        return next_local_map

    def create_mappredictor_input(self, local_map, action):
        local_map = local_map.cpu()
        b = local_map.size(0)
        mappredictor_input = []
        for i in range(b):
            map = local_map[i, ...]
            #plt.subplot(1, 2, 1)
            #map_vis = self.visualize_map(map)
            #plt.imshow(map_vis)
            if torch.is_tensor(action):
                action = action.cpu()
                a = action[i, ...]
                next_x = 64 + a[0][0] * self.time_step
                next_y = 64 + a[0][1] * self.time_step
            else:
                next_x = 64 + action.vx * self.time_step
                next_y = 64 + action.vy * self.time_step
            #plt.plot(64, 64, 'o')
            #plt.plot(next_x, next_y, 'o')
            map = self.get_local_map(map, (int(next_y), int(next_x)))
            #plt.subplot(1, 2, 2)
            #plt.plot(64, 64, 'o')
            #map_vis = self.visualize_map(map)
            #plt.imshow(map_vis)
            #plt.show()

            mappredictor_input.append(map)
        output = mappredictor_input[0].unsqueeze(0)
        for i in range(1, b):
            output = torch.cat([output, mappredictor_input[i].unsqueeze(0)], dim=0)
        return output.to('cuda:0')

    def visualize_map(self, semmap, bg=1.0):
        n_cat = semmap.shape[0] - 2 # Exclude floor and wall
        def compress_semmap(semmap):
            c_map = np.zeros((semmap.shape[1], semmap.shape[2]))
            for i in range(semmap.shape[0]):
                c_map[semmap[i] > 0.] = i+1
            return c_map

        palette = [
            int(bg * 255), int(bg * 255), int(bg * 255), # Out of bounds
            230, 230, 230, # Free space
            77, 77, 77, # Obstacles
        ]
        palette += [c for color in d3_40_colors_rgb[:n_cat] for c in color.tolist()]
        semmap = asnumpy(semmap)
        c_map = compress_semmap(semmap)
        semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
        semantic_img.putpalette(palette)
        semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = np.array(semantic_img)

        return semantic_img

    def crop_map(self, h, x, crop_size, mode="bilinear"):
        """
        Crops a tensor h centered around location x with size crop_size
        Inputs:
            h - (bs, F, H, W)
            x - (bs, 2) --- (x, y) locations
            crop_size - scalar integer
        Conventions for x:
            The origin is at the top-left, X is rightward, and Y is downward.
        """

        bs, _, H, W = h.size()
        Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
        Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
        start = -(crop_size - 1) / 2 if crop_size % 2 == 1 else -(crop_size // 2)
        end = start + crop_size - 1
        x_grid = (
            torch.arange(start, end + 1, step=1)
                .unsqueeze(0)
                .expand(crop_size, -1)
                .contiguous()
                .float()
        )
        y_grid = (
            torch.arange(start, end + 1, step=1)
                .unsqueeze(1)
                .expand(-1, crop_size)
                .contiguous()
                .float()
        )
        center_grid = torch.stack([x_grid, y_grid], dim=2).to(
            h.device
        )  # (crop_size, crop_size, 2)

        x_pos = x[:, 0] - Wby2  # (bs, )
        y_pos = x[:, 1] - Hby2  # (bs, )

        crop_grid = center_grid.unsqueeze(0).expand(
            bs, -1, -1, -1
        )  # (bs, crop_size, crop_size, 2)
        crop_grid = crop_grid.contiguous()

        # Convert the grid to (-1, 1) range
        crop_grid[:, :, :, 0] = (
                                        crop_grid[:, :, :, 0] + x_pos.unsqueeze(1).unsqueeze(2)
                                ) / Wby2
        crop_grid[:, :, :, 1] = (
                                        crop_grid[:, :, :, 1] + y_pos.unsqueeze(1).unsqueeze(2)
                                ) / Hby2

        h_cropped = F.grid_sample(h, crop_grid, mode=mode, align_corners=False)

        return h_cropped

    def get_local_map(self, in_semmap, location):
        y, x = location
        y, x = int(y), int(x)
        crop_center = torch.Tensor([[x, y]])
        map_ = self.crop_map(in_semmap.unsqueeze(0), crop_center, 960)
        _, N, H, W = map_.shape
        map_center = torch.Tensor([[W / 2.0, H / 2.0]])
        local_map = self.crop_map(map_, map_center, 128, 'nearest').squeeze(0)

        return local_map


class LinearStatePredictor(object):
    def __init__(self, config, time_step):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = False
        self.kinematics = config.action_space.kinematics
        self.time_step = time_step

    def __call__(self, state, action):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        next_robot_state = self.compute_next_state(state[0], action)
        next_human_states = self.linear_motion_approximator(state[1])

        next_observation = [next_robot_state, next_human_states]
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

    @staticmethod
    def linear_motion_approximator(human_states):
        """ approximate human states with linear motion, input shape : (batch_size, human_num, human_state_size)
        """
        # px, py, vx, vy, radius
        next_state = human_states.clone().squeeze()
        next_state[:, 0] = next_state[:, 0] + next_state[:, 2]
        next_state[:, 1] = next_state[:, 1] + next_state[:, 3]

        return next_state.unsqueeze(0)

