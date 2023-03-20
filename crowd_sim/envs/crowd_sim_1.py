import logging
import random
import math

import gym
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import tensor_to_joint_state, JointState
from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
import matplotlib.patches as mpathes

import os
import tqdm
import argparse
import multiprocessing as mp
from poni.default import get_cfg
from poni.dataset_1 import SemanticMapDataset
import matplotlib.pyplot as plt
import cv2
import numpy
from poni.constants import d3_40_colors_rgb
from einops import asnumpy, repeat
from PIL import Image, ImageFont, ImageDraw

ACTIVE_DATASET = "mp3d"
DATASET = ACTIVE_DATASET
OUTPUT_MAP_SIZE = 24.0
MASKING_MODE = "spath"
MASKING_SHAPE = "square"

SEED = 123
DATA_ROOT = "/home/cbl/RelationalGraphLearning_staticobs+stgcnn/crowd_nav/semantic_maps"
FMM_DISTS_SAVED_ROOT = "/home/cbl/RelationalGraphLearning_staticobs+stgcnn/crowd_nav/fmm_dists_123"
NUM_SAMPLES = {'train': 440, 'val': 105}
SAVE_ROOT = "/home/cbl/RelationalGraphLearning/crowd_nav/precomputed_dataset_mp3d_24.0_123_spath"

classes_SSCNav = ['wall', 'floor', 'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa',
                  'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs', 'ceiling', 'toilet', 'stool',
                  'towel', 'mirror', 'tv_monitor', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting',
                  'beam', 'railing', 'shelving', 'blinds', 'gym_equipment', 'seating', 'board_panel', 'furniture',
                  'appliances', 'clothes', 'objects', 'misc']

classes_PONI = ["floor", "wall", "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed", "chest_of_drawers",
                "plant", "sink", "toilet", "stool", "towel", "tv_monitor", "shower", "bathtub", "counter", "fireplace",
                "gym_equipment", "seating", "clothes"]


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        self.phase = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.distance_factor = config.reward.distance_factor
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        #这里config文件中的参数是orca
        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            #将centralized_planner设置为CentralizedORCA
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num
        #self.obstacle_num = config.sim.obstacle_num

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def get_approx(self, img, contours, length_p=0.1):
        """获取逼近多边形

        :param img: 处理图片
        :param contour: 连通域
        :param length_p: 逼近长度百分比
        """
        img_adp = img.copy()
        #print(len(contours))
        all_approx = []
        cnt = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > 20:
                # 逼近长度计算
                epsilon = length_p * cv2.arcLength(contours[i], True)
                # 获取逼近多边形
                approx = cv2.approxPolyDP(contours[i], epsilon, True)
                #print(approx)
                #cnt += numpy.size(approx)
                # 绘制显示多边形
                all_approx = all_approx + [approx]
                #cv2.drawContours(img_adp, [approx], 0, (0, 0, 255), 2)
        #print("point size:", cnt)
        #cv2.imshow("result %.5f" % length_p, img_adp)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return all_approx

    def  generate_dataset(self, kwargs):
        cfg = kwargs["cfg"]
        split = kwargs["split"]
        name = kwargs["name"]

        dataset = SemanticMapDataset(
            cfg.DATASET, split=split, scf_name=name, seed=SEED
        )
        #name, semmap, fmm_dists, map_xyz_info, nav_space, nav_locs = dataset.get_item_by_name(name)
        return name, dataset

    def precompute_dataset_for_map(self, level):

        channel = self.semmap[level+1, ...].astype(numpy.uint8) * 255  # free:0, Occ:255
        vis = cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)
        img_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        if level==0:
            edges = cv2.Canny(img_gray, 50, 200)
            minLineLength = 15  # 1最低线段的长度，小于这个值的线段被抛弃
            maxLineGap = 10  # 线段中点与点之间连接起来的最大距离，在此范围内才被认为是单行
            #n,1,4包含每条线段的两个顶点
            lines = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 30, minLineLength=minLineLength, maxLineGap=maxLineGap)
            return lines
        else:
            _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            obs = self.get_approx(vis, contours, 0.01)
            return obs

    def precompute_dataset(self):
        cfg = get_cfg()
        cfg.defrost()
        cfg.SEED = SEED
        cfg.DATASET.dset_name = DATASET
        cfg.DATASET.root = DATA_ROOT
        cfg.DATASET.output_map_size = OUTPUT_MAP_SIZE
        cfg.DATASET.fmm_dists_saved_root = FMM_DISTS_SAVED_ROOT
        cfg.DATASET.masking_mode = MASKING_MODE
        cfg.DATASET.masking_shape = MASKING_SHAPE
        cfg.DATASET.visibility_size = 3.0  # m
        cfg.freeze()

        os.makedirs(SAVE_ROOT, exist_ok=True)

        split = "train"
        os.makedirs(os.path.join(SAVE_ROOT, split), exist_ok=True)
        dataset = SemanticMapDataset(cfg.DATASET, split=split)
        n_maps = len(dataset)
        #print('Maps', n_maps)
        n_samples_per_map = (NUM_SAMPLES[split] // n_maps)
        #print(n_samples_per_map)
        #print(n_maps)

        map_id=-1
        map_id_range = None
        num_workers = 1
        if map_id != -1:
            map_names = [dataset.names[map_id]]
        elif map_id_range is not None:
            assert len(map_id_range) == 2
            map_names = [
                dataset.names[i]
                for i in range(map_id_range[0], map_id_range[1] + 1)
            ]
        else:
            map_names = dataset.names

        pool = mp.Pool(processes=num_workers)
        inputs = []
        for name in map_names:
            kwargs = {
                "cfg": cfg,
                "split": "train",
                "name": name,
                "n_samples_per_map": n_samples_per_map,
                "save_root": f'{SAVE_ROOT}/{split}',
            }
            inputs.append(kwargs)
        return inputs

    def set_robot(self, robot):
        self.robot = robot   

    def generate_robot(self):
        ys, xs = self.nav_locs
        random.seed()
        rnd_ix = random.randint(0, xs.shape[0] - 1)
        start_x, start_y = xs[rnd_ix], ys[rnd_ix]

        rnd_ix = random.randint(0, xs.shape[0] - 1)
        goal_x, goal_y = xs[rnd_ix], ys[rnd_ix]


        return start_x, start_y, goal_x, goal_y   

    def generate_human(self, human=None):
        if human is None:
            #根据config文件配置human的参数
            human = Human(self.config, 'humans')
        #配置文件里面设置不更改行人的首选速度和半径
        if self.randomize_attributes:
            human.sample_random_attributes()

        if self.current_scenario == 'circle_crossing':
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * human.v_pref
                py_noise = (np.random.random() - 0.5) * human.v_pref
                #生成的行人有噪声地分布在以circle_radius为半径的圆上
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    #min_dist是新生成的这个人和之前环境中所有智能体的最小距离
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    #新生成的这个人的位置不能离之前生成智能体的位置太近，也不能里之前生成智能体的目标位置太近
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            #添加噪声让行人分布在一个圆上，目标位置就是起始位置的对角线位置，初始速度设置为0，theta也设置为0
            human.set(px, py, -px, -py, 0, 0, 0)

        elif self.current_scenario == 'square_crossing':
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            while True:
                px = np.random.random() * self.square_width * 0.5 * sign
                py = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                gx = np.random.random() * self.square_width * 0.5 * - sign
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, gx, gy, 0, 0, 0)

        elif self.current_scenario == 'from_data':
            ys, xs = self.nav_locs
            random.seed()
            while True:
                rnd_ix = random.randint(0, xs.shape[0] - 1)
                start_x, start_y = xs[rnd_ix], ys[rnd_ix]
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((start_x - agent.px, start_y - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                rnd_ix = random.randint(0, xs.shape[0] - 1)
                goal_x, goal_y = xs[rnd_ix], ys[rnd_ix]
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((goal_x - agent.gx, goal_y - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break

            print(start_x, start_y, goal_x, goal_y)
            human.set(start_x, start_y, goal_x, goal_y, 0, 0, 0)

        return human

    def generate_human_start(self, human=None):
        if human is None:
            human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        ys, xs = self.nav_locs
        random.seed()
        while True:
            rnd_ix = random.randint(0, xs.shape[0] - 1)
            start_x, start_y = xs[rnd_ix], ys[rnd_ix]
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((start_x - agent.px, start_y - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            rnd_ix = random.randint(0, xs.shape[0] - 1)
            goal_x, goal_y = xs[rnd_ix], ys[rnd_ix]
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((goal_x - agent.gx, goal_y - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break

        #print(start_x, start_y, goal_x, goal_y)
        human.set(start_x, start_y, goal_x, goal_y, 0, 0, 0)

        return human

    #生成正方形障碍物,生成正方形障碍物的中心坐标，该障碍物的长宽都为1，且分别平行于x, y轴
    def generate_obstacle(self, i, obstacle=None):
        if self.current_scenario == 'circle_crossing':
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5)
                py_noise = (np.random.random() - 0.5)
                px = self.circle_radius * np.cos(angle) / 2 + px_noise
                #print(px)
                py = self.circle_radius * np.sin(angle) / 2 + py_noise
                #print(py)
                collide = False
                '''
                for obs in self.obstacles:
                    if abs(px-obs[0]) < 1 or abs(py-obs[1]) < 1:
                        collide = True
                        break
                '''
                if not collide:
                    break

        elif self.current_scenario == 'square_crossing':
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            while True:
                px = np.random.random() * self.square_width * 0.5 * sign
                py = (np.random.random() - 0.5) * self.square_width
                collide = False
                for obs in self.obstacles:
                    if abs(px-obs[0]) < 1 or abs(py-obs[1]) < 1:
                        collide = True
                        break
                if not collide:
                    break

        elif self.current_scenario == 'from_data':

            obs = self.precompute_dataset_for_map(i)

        return obs

    #为当前环境中的机器人生成起始位置，目标位置，初速度，theta;然后按照config文件中的人数生成分布在圆或者正方形形状上的行人，为他们设置位置，速度，目标位置
    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                     'val': 0, 'test': self.case_capacity['val']}

        #self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        kwargs = self.precompute_dataset()
        name, dataset = self.generate_dataset(kwargs[0])
        name, self.semmap, _, _, nav_space, self.nav_locs = dataset.get_item_by_name(name)
        robot_px, robot_py, robot_gx, robot_gy = self.generate_robot()
        self.robot.set(robot_px, robot_py, robot_gx, robot_gy, 0, 0, np.pi / 2)
        if self.case_counter[phase] >= 0:
            np.random.seed(base_seed[phase] + self.case_counter[phase])
            random.seed(base_seed[phase] + self.case_counter[phase])
            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
            if not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                # only CADRL trains in circle crossing simulation
                human_num = 1
                self.current_scenario = 'circle_crossing'
            else:
                self.current_scenario = self.test_scenario
                human_num = self.human_num
                #obstacle_num = self.obstacle_num
                obstacle_num = 22
            self.humans = []
            #行人的生成是用障碍物数据集中间的安全区域生成
            for _ in range(human_num):
                self.humans.append(self.generate_human_start())

            self.obstacles = []

            #在这里生成障碍物，但是用数据集生成，obstacles的第一维是障碍物的语义信息
            #obstacle_num的含义表示障碍物的层数
            for i in range(obstacle_num):
                #生成第i层障碍物
                self.obstacles.append(self.generate_obstacle(i))

            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        if hasattr(self.robot.policy, 'get_matrix_A'):
            self.As = list()
        if hasattr(self.robot.policy, 'get_feat'):
            self.feats = list()
        if hasattr(self.robot.policy, 'get_X'):
            self.Xs = list()
        if hasattr(self.robot.policy, 'trajs'):
            self.trajs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob,ob_nov = self.compute_observation_for(self.robot)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob,ob_nov,self.obstacles

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    #step里面的工作主要是计算行人在当前时刻采取的动作，并利用所有智能体的位置和动作计算是否有碰撞发生，如果没有碰撞发生则计算下一时刻机器人的位置，速度则更改为当前时刻的动作
    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        random.seed()
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states, self.obstacles)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states, self.obstacles)
        else:
            human_actions = []
            for human in self.humans:
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob, self.obstacles))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.debug("Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(human.id, closest_dist, self.global_time))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < 2.0   #self.robot.radius   #距离目标小于2米认为到达

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Discomfort(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'get_matrix_A'):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, 'get_feat'):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, 'get_X'):
                self.Xs.append(self.robot.policy.get_X())
            if hasattr(self.robot.policy, 'traj'):
                self.trajs.append(self.robot.policy.get_traj())

            # update all agents
            self.robot.step(action)
            i=0
            for human, action in zip(self.humans, human_actions):
                human.step(action)
                #print(human.vx, human.vy)
                if abs(human.vx) < 0.01 and abs(human.vy) < 0.01:
                    ys, xs = self.nav_locs
                    rnd_ix = random.randint(0, xs.shape[0] - 1)
                    #print(rnd_ix)
                    goal_x, goal_y = xs[rnd_ix], ys[rnd_ix]
                    gx, gy = human.get_goal_position()
                    self.humans[i].set(gx, gy, goal_x, goal_y, 0, 0, 0)

                if self.nonstop_human and human.reached_destination():
                    ys, xs = self.nav_locs
                    rnd_ix = random.randint(0, xs.shape[0] - 1)
                    #print(rnd_ix)
                    goal_x, goal_y = xs[rnd_ix], ys[rnd_ix]
                    gx, gy = human.get_goal_position()
                    self.humans[i].set(gx, gy, goal_x, goal_y, 0, 0, 0)
                    #self.humans.append(self.generate_human(human))
                i=i+1

            self.global_time += self.time_step
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob, ob_nov = self.compute_observation_for(self.robot)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, ob_nov, reward, done, info

    def only_ped_step(self):
        random.seed()
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states, self.obstacles)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states, self.obstacles)
        else:
            human_actions = []
            for human in self.humans:
                ob, _ = self.compute_observation_for(human)
                human_actions.append(human.act(ob, self.obstacles))
    
        # collision detection between humans
        #为什么人和人之间的碰撞只需要简单计算开始时刻没有发生碰撞就行了
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        for human, action in zip(self.humans, human_actions):
                human.step(action)
                #这里默认是false
                if self.nonstop_human and human.reached_destination():
                    self.generate_human(human)
        i=0
        for human, action in zip(self.humans, human_actions):
            human.step(action)
            #print(human.vx, human.vy)
            if abs(human.vx) < 0.01 and abs(human.vy) < 0.01:
                ys, xs = self.nav_locs
                rnd_ix = random.randint(0, xs.shape[0] - 1)
                #print(rnd_ix)
                goal_x, goal_y = xs[rnd_ix], ys[rnd_ix]
                gx, gy = human.get_goal_position()
                self.humans[i].set(gx, gy, goal_x, goal_y, 0, 0, 0)

            if self.nonstop_human and human.reached_destination():
                ys, xs = self.nav_locs
                rnd_ix = random.randint(0, xs.shape[0] - 1)
                #print(rnd_ix)
                goal_x, goal_y = xs[rnd_ix], ys[rnd_ix]
                gx, gy = human.get_goal_position()
                self.humans[i].set(gx, gy, goal_x, goal_y, 0, 0, 0)
                #self.humans.append(self.generate_human(human))
            i=i+1


        ob, ob_nov = self.compute_observation_for(self.robot)
        return ob, ob_nov


    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            #print('s')
            for human in self.humans:
                ob.append(human.get_observable_state_il())
                #print(human.get_observable_state().px, human.get_observable_state().py, human.get_observable_state().vx, human.get_observable_state().vy)
        else:
            ob = [other_human.get_observable_state_il() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state_il()]

        if agent == self.robot:
            ob_nov = []
            #print('s')
            for human in self.humans:
                ob_nov.append(human.get_observable_state())
                #print(human.get_observable_state().px, human.get_observable_state().py, human.get_observable_state().vx, human.get_observable_state().vy)
        else:
            ob_nov = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob_nov += [self.robot.get_observable_state()]
        return ob, ob_nov

    def visualize_map(self, semmap, bg=1.0, dataset='gibson'):
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
        if dataset == 'gibson':
            palette += [int(x * 255.) for x in gibson_palette[15:]]
        else:
            palette += [c for color in d3_40_colors_rgb[:n_cat]
                        for c in color.tolist()]
        semmap = asnumpy(semmap)
        c_map = compress_semmap(semmap)
        semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
        semantic_img.putpalette(palette)
        semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = np.array(semantic_img)

        return semantic_img

    def render(self, mode='video', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('hsv', 15)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True
        obstacle_color=['#1F77B4', '#BEC7E8', '#FF7F0E', '#FFBB78', '#2CA02C', '#98DF8A', '#CEDB9C', '#8C6D31', '#9467BD', '#C5B0D5', '#8C564B',
                        '#C49C90', '#E377B2', '#F7B6D2', '#7F7F7F', '#C7C7C7', '#BCBD22', '#DBDB8D', '#17BECF', '#9EDAE5', '#393B79', '#5254A3',
                        '#6B6ECF', '#9C9EDE']

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=15)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=15)
                ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=False, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)
            #ax.set_xlim(-11, 11)
            #ax.set_ylim(-11, 11)
            ax.set_xlim(0, 600)
            ax.set_ylim(0, 600)
            ax.set_xlabel('x(0.05m/pixel)', fontsize=14)
            ax.set_ylabel('y(0.05m/pixel)', fontsize=14)
            show_human_start_goal = False                  #True时可视化行人的起点和终点
            show_semantic_map = True                       #True时可视化语义地图
            show_wall_and_bounding_box = False             #True时可视化wall和物体边界框框


            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            if show_human_start_goal:
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i],
                                               marker='*', linestyle='None', markersize=8)
                    ax.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=8)
                    ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8)
            robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
            ax.add_artist(robot_start)
            
            #绘制障碍物
            '''
            xy=[]
            rect=[]
            for i, obs in enumerate(self.obstacles):
                #print(obs[0],obs[1])
                xy.append(np.array([obs[0]-0.5, obs[1]-0.5]))
                rect.append(mpathes.Rectangle(xy[i], 1, 1, color='r'))
                ax.add_patch(rect[i])
            '''
            if show_semantic_map:
                local_map = self.visualize_map(self.semmap[0:23, ...], dataset="mp3d") 
                plt.imshow(local_map)
            
            if show_wall_and_bounding_box:
                for level in range(len(self.obstacles)):                    
                    if level == 0:
                        for i in range(len(self.obstacles[level])):
                            plt.plot([self.obstacles[level][i,0,0], self.obstacles[level][i,0,2]], [self.obstacles[level][i,0,1],self.obstacles[level][i,0,3]], color=obstacle_color[level+1])
                    else:
                        #print(self.obstacles[level])
                        for i in range(len(self.obstacles[level])):#这里是同语义障碍物的个数
                            for j in range(len(self.obstacles[level][i])-1):#某一个障碍物中点的个数
                                plt.plot([self.obstacles[level][i][j,0,0], self.obstacles[level][i][j+1,0,0]], [self.obstacles[level][i][j,0,1], self.obstacles[level][i][j+1,0,1]], color=obstacle_color[level+1])
                                #print([self.obstacles[level][i][j,0,0]],' ',[self.obstacles[level][i][j,0,1]])
                            plt.plot([self.obstacles[level][i][0,0,0], self.obstacles[level][i][-1,0,0]], [self.obstacles[level][i][0,0,1], self.obstacles[level][i][-1,0,1]], color=obstacle_color[level+1])
            
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')

            circle_radius_ = 5               #行人和机器人可视化半径
            robot = plt.Circle(robot_positions[0], circle_radius_, fill=False, color=robot_color)
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot_start, goal], ['Start', 'Goal'], fontsize=14)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], circle_radius_, fill=False, color=cmap(i))
                      for i in range(len(self.humans))]

            # disable showing human numbers
            if display_numbers:
                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                          color='black') for i in range(len(self.humans))]
            robot_marker = plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset, 'Robot', color='black')
            ax.add_artist(robot_marker)

            for i, human in enumerate(humans):
                ax.add_artist(human)
                if display_numbers:
                    ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(0.4, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(time)

            # visualize attention scores
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            orientations = []
            for i in range(self.human_num + 1):
                orientation = []
                for state in self.states:
                    agent_state = state[0] if i == 0 else state[1][i - 1]
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        direction = (
                        (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                           agent_state.py + radius * np.sin(agent_state.theta)))
                    else:
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                        agent_state.py + radius * np.sin(theta)))
                    orientation.append(direction)
                orientations.append(orientation)
                if i == 0:
                    arrow_color = 'black'
                    arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)]
                else:
                    arrows.extend(
                        [patches.FancyArrowPatch(*orientation[0], color=human_colors[i - 1], arrowstyle=arrow_style)])

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            if len(self.trajs) != 0:
                human_future_positions = []
                human_future_circles = []
                for traj in self.trajs:
                    human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
                                              for step in range(self.robot.policy.planning_depth)]
                                             for i in range(self.human_num)]
                    human_future_positions.append(human_future_position)

                for i in range(self.human_num):
                    circles = []
                    for j in range(self.robot.policy.planning_depth):
                        circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
                        ax.add_artist(circle)
                        circles.append(circle)
                    human_future_circles.append(circles)

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                robot_marker.set_position((robot.center[0] - x_offset, robot.center[1] + y_offset))

                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    if display_numbers:
                        human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] + y_offset))
                for arrow in arrows:
                    arrow.remove()

                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color='black',
                                                          arrowstyle=arrow_style)]
                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=cmap(i - 1),
                                                               arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                    # if hasattr(self.robot.policy, 'get_attention_weights'):
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                if len(self.trajs) != 0:
                    for i, circles in enumerate(human_future_circles):
                        for j, circle in enumerate(circles):
                            circle.center = human_future_positions[global_step][i][j]

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        if hasattr(self.robot.policy, 'action_values'):
                            plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 100)
            anim.running = True

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()
        else:
            raise NotImplementedError
