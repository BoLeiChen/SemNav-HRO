import logging
import torch
import numpy as np
from numpy.linalg import norm
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_nav.policy.state_predictor import StatePredictor, LinearStatePredictor, MapPredictor
from crowd_nav.policy.stgcnn_model import social_stgcnn, Social_StatePredictor
from crowd_nav.policy.graph_model import RGL, GAT_RL
from crowd_nav.policy.value_estimator import ValueEstimator
import matplotlib.pyplot as plt
import random
import datetime

class ModelPredictiveRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ModelPredictiveRL'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.action_space = None
        self.rotation_constraint = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.robot_state_dim = 7
        self.human_state_dim = 3
        self.v_pref = 2
        self.share_graph_model = None
        self.value_estimator = None
        self.linear_state_predictor = None
        #用一个新的predictor替代原来的这个轨迹预测模型
        self.state_predictor = None
        self.map_predictor = None
        #lsy:star
        #在这里定义social-stgcnn需要的一系列参数
        self.n_stgcnn = 1
        self.n_txpcnn = 3
        self.kernel_size = 3
        self.obs_seq_len = 3
        self.pred_seq_len = 2
        self.output_size = 5
        #lsy:end
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.sparse_search = None
        self.sparse_speed_samples = 2
        self.sparse_rotation_samples = 8
        self.action_group_index = []
        self.traj = None
        self.lsy = True
        self.use_local_map_semantic = None
        self.use_RGL = None
        self.RGL_origin = None

    #初始化model_predictive_rl模型后，实际上就构建了RGL关系图来预测状态价值函数以及行人的状态，这里不使用线性状态预测器且价值函数和行人状态不共享图神经网络参数
    def configure(self, config):
        self.set_common_parameters(config)
        self.use_RGL = config.use_Ours
        self.use_local_map_semantic = config.use_local_map_semantic
        self.RGL_origin = config.RGL_origin

        self.planning_depth = config.model_predictive_rl.planning_depth
        self.do_action_clip = config.model_predictive_rl.do_action_clip
        if hasattr(config.model_predictive_rl, 'sparse_search'):
            self.sparse_search = config.model_predictive_rl.sparse_search
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.linear_state_predictor = config.model_predictive_rl.linear_state_predictor

        if self.linear_state_predictor:
            self.state_predictor = LinearStatePredictor(config, self.time_step)
            graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_estimator = ValueEstimator(config, graph_model, self.use_local_map_semantic)
            self.model = [graph_model, self.value_network]
        #执行这个分支，轨迹预测模型换成social-stgcnn
        #lsy:start
        elif self.lsy:
            if self.use_RGL:
                graph_model1 = RGL(config, self.robot_state_dim, self.human_state_dim)
            else:
                graph_model1 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
            self.value_estimator = ValueEstimator(config, graph_model1, self.use_local_map_semantic, self.use_RGL)
            if self.use_RGL:
                graph_model2 = RGL(config, self.robot_state_dim, self.human_state_dim)
            else:
                graph_model2 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
            self.state_predictor = Social_StatePredictor(graph_model2, config, self.time_step, n_stgcnn =self.n_stgcnn,n_txpcnn=self.n_txpcnn,
                output_feat=self.output_size,seq_len=self.obs_seq_len,
                kernel_size=self.kernel_size,pred_seq_len=self.pred_seq_len, RGL_origin = self.RGL_origin).cuda()
            self.map_predictor = MapPredictor(self.time_step, self.use_local_map_semantic)
            self.model = [graph_model1, graph_model2, self.value_estimator.value_network, self.value_estimator.mapconv,
            self.value_estimator.weight, self.state_predictor.human_motion_predictor, self.map_predictor.mappredictor]
        #执行这个分支，不使用线性状态预测器  
        #lsy:end
        else:
            if self.share_graph_model:
                graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = ValueEstimator(config, graph_model, self.use_local_map_semantic)
                self.state_predictor = StatePredictor(config, graph_model, self.time_step)
                self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]
            #执行这个分支
            else:
                graph_model1 = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = ValueEstimator(config, graph_model1, self.use_local_map_semantic, self.use_RGL)
                graph_model2 = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.state_predictor = StatePredictor(config, graph_model2, self.time_step, self.use_RGL)
                self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                              self.state_predictor.human_motion_predictor]

        logging.info('Planning depth: {}'.format(self.planning_depth))
        logging.info('Planning width: {}'.format(self.planning_width))
        logging.info('Sparse search: {}'.format(self.sparse_search))

        if self.planning_depth > 1 and not self.do_action_clip:
            logging.warning('Performing d-step planning without action space clipping!')

    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.state_predictor.time_step = time_step
        self.map_predictor.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_estimator

    def get_state_dict(self):
        if self.state_predictor.trainable and self.map_predictor.trainable:
            if self.lsy:
                print('loadlsy')
                return{
                    'graph_model1': self.value_estimator.graph_model.state_dict(),
                    'graph_model2': self.state_predictor.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict(),
                    'map_embedding': self.value_estimator.mapconv.state_dict(),
                    'value_weight': self.value_estimator.weight.state_dict(),
                    'motion_predictor_RGL': self.state_predictor.human_motion_predictor_RGL.state_dict(),
                    'motion_predictor': self.state_predictor.human_motion_predictor.state_dict(),
                    'map_predictor': self.map_predictor.mappredictor.state_dict()
                }
            else:
                if self.share_graph_model:
                    return {
                        'graph_model': self.value_estimator.graph_model.state_dict(),
                        'value_network': self.value_estimator.value_network.state_dict(),
                        'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                    }
                else:
                    return {
                        'graph_model1': self.value_estimator.graph_model.state_dict(),
                        'graph_model2': self.state_predictor.graph_model.state_dict(),
                        'value_network': self.value_estimator.value_network.state_dict(),
                        'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                    }
        else:
            return {
                    'graph_model': self.value_estimator.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict()
                }

    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):
        if self.state_predictor.trainable and self.map_predictor.trainable:
            if self.lsy:
                self.value_estimator.graph_model.load_state_dict(state_dict['graph_model1'])
                self.state_predictor.graph_model.load_state_dict(state_dict['graph_model2'])
                self.value_estimator.value_network.load_state_dict(state_dict['value_network'])
                self.value_estimator.mapconv.load_state_dict(state_dict['map_embedding'])
                self.value_estimator.weight.load_state_dict(state_dict['value_weight'])
                self.state_predictor.human_motion_predictor_RGL.load_state_dict(state_dict['motion_predictor_RGL'])
                self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])
                self.map_predictor.mappredictor.load_state_dict(state_dict['map_predictor'])
            else:
                if self.share_graph_model:
                    self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
                else:
                    self.value_estimator.graph_model.load_state_dict(state_dict['graph_model1'])
                    self.state_predictor.graph_model.load_state_dict(state_dict['graph_model2'])

                self.value_estimator.value_network.load_state_dict(state_dict['value_network'])
                self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])
        else:
            self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for j, speed in enumerate(speeds):
            if j == 0:
                # index for action (0, 0)
                self.action_group_index.append(0)
            # only two groups in speeds
            if j < 3:
                speed_index = 0
            else:
                speed_index = 1

            for i, rotation in enumerate(rotations):
                rotation_index = i // 2

                action_index = speed_index * self.sparse_rotation_samples + rotation_index
                self.action_group_index.append(action_index)

                if holonomic:
                    action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
                else:
                    action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def new_predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            #有epsilon的概率是随机选择一个action作为下一步的action 
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_action = None
            max_value = float('-inf')
            max_traj = None

            if self.do_action_clip:
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                action_space_clipped = self.action_clip(state_tensor, self.action_space, self.planning_width)
            else:
                action_space_clipped = self.action_space

            for action in action_space_clipped:
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                next_state = self.state_predictor(state_tensor, action)
                #这是d步方法，返回当前状态最大的回报，和预测出的未来的最优轨迹
                max_next_return, max_next_traj = self.V_planning(next_state, self.planning_depth, self.planning_width)
                #根据获得的动作来计算收益，因为奖励和是否发生碰撞，是否在安全区域里面有关，这都和行人的动作有关
                reward_est = self.estimate_reward(state, action)
                value = reward_est + self.get_normalized_gamma() * max_next_return
                #计算value来选择一个value最大的动作
                if value > max_value:
                    max_value = value
                    max_action = action
                    max_traj = [(state_tensor, action, reward_est)] + max_next_traj
            if max_action is None:
                max_action = action_space_clipped[np.random.choice(len(action_space_clipped))]
                #raise ValueError('Value network is not well trained.')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        else:
            self.traj = max_traj

        return max_action

    def predict(self, state_last, state, cur_memory, local_map = None):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)

        local_map = local_map.unsqueeze(0)
        if self.device == torch.device('cuda:0'):
            local_map = local_map.cuda()
        elif self.device is not None:
            local_map.to(self.device)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            #print(2)
            #有epsilon的概率是随机选择一个action作为下一步的action 
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_action = None
            max_q_map = None
            max_value = float('-inf')
            max_traj = None

            if self.do_action_clip:
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                action_space_clipped = self.action_clip(state_tensor, local_map, self.action_space, self.planning_width)
            else:
                action_space_clipped = self.action_space
            #要收集之前的8个步长才能预测出未来的步长
            for action in action_space_clipped:
                time1 = datetime.datetime.now()
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                next_state = self.state_predictor(state_tensor, cur_memory, action, state.human_states[0].radius)
                time2 = datetime.datetime.now()
                #print("state_predictor:",(time2 - time1).microseconds)
                next_local_map = self.map_predictor(local_map, action)
                time3 = datetime.datetime.now()
                #print("map_predictor:",(time3 - time2).microseconds)
                next_local_map = (next_local_map == next_local_map.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
                max_next_return, max_next_traj, q_map = self.V_planning(next_state, next_local_map, self.planning_depth, self.planning_width)
                time4 = datetime.datetime.now()
                #print("V_planning:",(time4 - time3).microseconds)
                #根据获得的动作来计算收益，因为奖励和是否发生碰撞，是否在安全区域里面有关，这都和行人的动作有关
                reward_est = self.estimate_reward(state_last, state, local_map, action)
                time5 = datetime.datetime.now()
                #print("estimate_reward:",(time5 - time4).microseconds)
                value = reward_est + self.get_normalized_gamma() * max_next_return

                #计算value来选择一个value最大的动作
                if value > max_value:
                    max_value = value
                    max_action = action
                    max_q_map = q_map
                    max_traj = [(state_tensor, action, reward_est)] + max_next_traj
                    #print('nx')
                    #print(next_state[0][0])
            if max_action is None:
                max_action = action_space_clipped[np.random.choice(len(action_space_clipped))]
                #raise ValueError('Value network is not well trained.')

        #plt.subplot(1, 1, 1)
        #plt.imshow(max_q_map)
        #plt.show()


        if self.phase == 'train':
            self.last_state = self.transform(state)
        else:
            self.traj = max_traj

        return max_action


    def action_clip(self, state, local_map, action_space, width, depth=1):
        values = []

        for action in action_space:
            next_state_est = self.state_predictor(state, action)
            next_local_map = self.map_predictor(local_map, action)
            next_local_map = (next_local_map == next_local_map.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
            next_return, _ = self.V_planning(next_state_est, next_local_map, depth, width)
            reward_est = self.estimate_reward(state, local_map, action)
            value = reward_est + self.get_normalized_gamma() * next_return
            values.append(value)

        if self.sparse_search:
            # self.sparse_speed_samples = 2
            # search in a sparse grained action space
            added_groups = set()
            max_indices = np.argsort(np.array(values))[::-1]
            clipped_action_space = []
            for index in max_indices:
                if self.action_group_index[index] not in added_groups:
                    clipped_action_space.append(action_space[index])
                    added_groups.add(self.action_group_index[index])
                    if len(clipped_action_space) == width:
                        break
        else:
            max_indexes = np.argpartition(np.array(values), -width)[-width:]
            clipped_action_space = [action_space[i] for i in max_indexes]

        # print(clipped_action_space)
        return clipped_action_space

    def V_planning(self, state, local_map, depth, width):
        """ Plans n steps into future. Computes the value for the current state as well as the trajectories
        defined as a list of (state, action, reward) triples

        """

        current_state_value, q_map = self.value_estimator(state, local_map)
        if depth == 1:
            return current_state_value, [(state, None, None)], q_map

        if self.do_action_clip:
            action_space_clipped = self.action_clip(state, local_map, self.action_space, width)
        else:
            action_space_clipped = self.action_space

        returns = []
        trajs = []

        for action in action_space_clipped:
            next_state_est = self.state_predictor(state, action)
            next_local_map = self.map_predictor(local_map, action)
            next_local_map = (next_local_map == next_local_map.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
            reward_est = self.estimate_reward(state, local_map, action)
            next_value, next_traj = self.V_planning(next_state_est, next_local_map, depth - 1, self.planning_width)
            return_value = current_state_value / depth + (depth - 1) / depth * (self.get_normalized_gamma() * next_value + reward_est)

            returns.append(return_value)
            trajs.append([(state, action, reward_est)] + next_traj)

        max_index = np.argmax(returns)
        max_return = returns[max_index]
        max_traj = trajs[max_index]

        return max_return, max_traj


    def estimate_reward(self, state_last, state, local_map, action):
        """ If the time step is small enough, it's okay to model agent as linear movement during this period

        """
        # collision detection
        if isinstance(state, list) or isinstance(state, tuple):
            state = tensor_to_joint_state(state)
        human_states = state.human_states
        robot_state = state.robot_state

        dmin = float('inf')
        collision = False
        #这里判断是否发生碰撞还要加上是否和障碍物发生碰撞
        for i, human in enumerate(human_states):
            px = human.px - robot_state.px
            py = human.py - robot_state.py
            if self.kinematics == 'holonomic':
                vx = (state_last.human_states[i].px-human.px)/self.time_step - action.vx
                vy = (state_last.human_states[i].py-human.py)/self.time_step - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + robot_state.theta)
                vy = human.vy - action.v * np.sin(action.r + robot_state.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - robot_state.radius
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between robot and obstacle
        if self.kinematics == 'holonomic':
            rx = robot_state.px + action.vx * self.time_step
            ry = robot_state.py + action.vy * self.time_step
        else:
            theta = robot_state.theta + action.r
            rx = robot_state.px + np.cos(theta) * action.v * self.time_step
            ry = robot_state.py + np.sin(theta) * action.v * self.time_step
        collision_with_obstacle = False
        map_x, map_y = self.env.occ_map.shape
        samplePoint = self.env.getRandomPointInCircle(100, robot_state.radius / 2 / 0.05, rx, ry)
        for i in range(len(samplePoint)):
            rx, ry = samplePoint[i][0], samplePoint[i][1]
            if rx >= map_y or rx < 0 or ry >= map_x or ry < 0:
                continue
            if self.env.occ_map[ry][rx] == 1:
                collision_with_obstacle = True
                break

        # check if reaching the goal
        if self.kinematics == 'holonomic':
            px = robot_state.px + action.vx * self.time_step
            py = robot_state.py + action.vy * self.time_step
        else:
            theta = robot_state.theta + action.r
            px = robot_state.px + np.cos(theta) * action.v * self.time_step
            py = robot_state.py + np.sin(theta) * action.v * self.time_step

        end_position = np.array((px, py))
        if self.env.objectnav:
            reaching_goal = self.success_checker_for_object_nav(end_position, 1)
        else:
            reaching_goal = norm(end_position - np.array([robot_state.gx, robot_state.gy])) < robot_state.radius

        dis_reaching_goal = norm(end_position - np.array([robot_state.gx, robot_state.gy]))

        if collision:
            reward = -0.5
        elif collision_with_obstacle:
            reward = -0.25
        elif reaching_goal:
            reward = 5
        elif dmin < 0.5:
            # adjust the reward based on FPS
            reward = (dmin - 0.5) * 0.5 #* self.time_step
        else:
            #reward = 0.1 - self.env.global_time / self.time_step * 0.005# + 1 / dis_reaching_goal
            reward = 1 / dis_reaching_goal - self.env.global_time / self.time_step * 0.005
        return reward

    def success_checker_for_object_nav(self, robot_pos, check_range):

        object_goal_layer = self.env.semmap[self.env.object_goal_idx, ...]

        success = False
        map_x, map_y = object_goal_layer.shape
        samplePoint = self.env.getRandomPointInCircle(200, check_range / 0.05, robot_pos[0], robot_pos[1])
        for i in range(len(samplePoint)):
            rx, ry = samplePoint[i][0], samplePoint[i][1]
            if rx >= map_y or rx < 0 or ry >= map_x or ry < 0:
                continue
            if object_goal_layer[ry][rx] == 1:
                success = True
                break
        return success


    def transform(self, state):
        """
        Take the JointState to tensors

        :param state:
        :return: tensor of shape (# of agent, len(state))
        """
        #把机器人和行人的状态变成张量
        robot_state_tensor = torch.Tensor([state.robot_state.to_tuple()]).to(self.device)
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in state.human_states]). \
            to(self.device)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        return robot_state_tensor, human_states_tensor
