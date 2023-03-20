import os
import logging
import copy
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.info import *
import numpy as np
from utils.memory import TrajectoryDataset

class Explorer(object):
    def __init__(self, use_RE3, env, robot, device, writer, memory=None, ped_memory=None, cur_memory=None, ped_memory_batch=None, gamma=None,
                 target_policy=None, intrinsic_reward = None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.ped_memory = ped_memory
        self.cur_memory = cur_memory
        self.ped_memory_batch = ped_memory_batch
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

        self.intrinsic_reward_alg = intrinsic_reward
        self.use_RE3 = use_RE3

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

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, update_ped_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        collisionobs_times = []
        timeout_times = []
        success = 0
        collision = 0
        collisionobs = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        collisionobs_cases = []
        timeout_cases = []

        if k != 1:
            pbar = tqdm(total=k)
        else:
            pbar = None

        #探索执行k次，每次达到结束条件则探索终止
        #如果是训练模式，得先让行人走8步这样才能保证每次预测有足够多的历史数据
        for i in range(k):
            #要修改reset函数
            ob,ob_nov,obs,local_map = self.env.reset(phase)
            ob_last=ob_nov#ob_last是上一个时刻的行人的状态，用两个时刻的状态才能判断是否发生碰撞
            #ped_ob = self.ob_ped_data(0,ob)
            done = False
            states = []
            #lsy:start
            #用于收集探索过程中每一步行人的状态
            ped_states = []
            ped_time = 0
            ped_states.extend(self.ob_ped_data(ped_time,ob))
            #lsy:end
            actions = []
            rewards = []
            local_maps = []
            if imitation_learning==False:
                cur_ped_states = []
                cur_ped_states.extend(self.ob_ped_data(ped_time,ob))
                for i in range(2):#观察3步轨迹数据还要加上reset那一步
                    ped_time = ped_time + 1
                    ob_last = ob_nov
                    ob,ob_nov = self.env.only_ped_step()
                    #self.cur_memory.onetime_push(self.ob_ped_data(ped_time,ob))
                    #用一个序列来记录20次的行人运动数据
                    cur_ped_states.extend(self.ob_ped_data(ped_time,ob)) 
                for i in range(2):#预测未来两步的轨迹数据
                    ped_time = ped_time + 1
                    cur_ped_states.extend(self.ob_ped_data(ped_time,ob))
                if self.cur_memory is None :
                    raise ValueError('Ped_Memory  is not set!')
                self.cur_memory.push(cur_ped_states)
                #print(np.array(cur_ped_states))
            #print(self.robot.gx, self.robot.gy)
            while not done:
                ped_time = ped_time + 1
                #lsy:start
                if imitation_learning:
                    action = self.robot.act(obs, self.cur_memory, ob, local_map=local_map, imitation_learning=imitation_learning)
                else:
                    #这个函数里面改成只需要ob1的样子，不需要速度
                    action = self.robot.act(obs, self.cur_memory, ob_nov, ob_last, local_map=local_map, imitation_learning=imitation_learning)
                #action = self.robot.act(ped_ob)
                #lsy:end
                #done表示的含义是探索是否结束
                #ob返回的是以class为子元素的序列
                ob_last = ob_nov
                ob, ob_nov, reward, done, info, local_map = self.env.step(action)
                if imitation_learning == False:
                    for i in range(len(ob)):
                        cur_ped_states.remove(cur_ped_states[0])
                    cur_ob_ped_data = self.ob_ped_data(ped_time-2,ob)
                    for i in range(len(ob)):
                        cur_ped_states[2*len(ob)+i] = cur_ob_ped_data[i]
                    #cur_ped_states.extend(self.ob_ped_data(ped_time,ob))
                    #print(np.array(cur_ob_ped_data).shape)
                    cur_ped_states.extend(self.ob_ped_data(ped_time,ob))
                    self.cur_memory.clear()
                    if done == False:
                        self.cur_memory.push(cur_ped_states)
                #返回的状态都是可被观测到的状态，且都是step之前的状态，所以结束状态就没有被记录
                #这里的last_state是JoinState的类，里面既包含了行人状态也包含了机器人状态，ob里面只包含了行人的状态
                states.append(self.robot.policy.last_state)
                #lsy:start
                #用列表的结构计算一次探索过程中的行人轨迹
                #ped_states = ped_states + self.ob_ped_data(ped_time,ob)
                ped_states.extend(self.ob_ped_data(ped_time,ob))
                #lsy:end
                actions.append(action)
                #reward是一个序列，里面记录的是每个时刻得到的奖励
                rewards.append(reward)
                local_maps.append(local_map)
              
            #video_file = "/home/cbl/RelationalGraphLearning_staticobs+stgcnn/crowd_nav/video/demo.mp4"
            video_file = None
            #self.env.render('video', video_file)


            if isinstance(info, Discomfort):
                discomfort += 1
                min_dist.append(info.min_dist)
            #分别计算这k次探索中成功，碰撞，超时的次数
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
                logging.info('Success !!!')
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
                logging.info('Collision with human !!!')
            elif isinstance(info, CollisionObs):
                collisionobs += 1
                collisionobs_cases.append(i)
                collisionobs_times.append(self.env.global_time)
                logging.info('Collision with obstacle !!!')
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
                logging.info('Timeout !!!')
            else:
                raise ValueError('Invalid end signal from environment')

            #在模仿学习过程中，memory的会被更新然后用于训练过程，每个回合都会更新一次
            if update_memory:
                #超时这种情况不用来更新memory
                if isinstance(info, ReachGoal) or isinstance(info, Collision) or isinstance(info, CollisionObs):
                    # only add positive(success) or negative(collision) experience in experience set
                    embeddings = []
                    if self.use_RE3:
                        states_tensor = []
                        if imitation_learning:
                            for s in states:
                                (r, h) = self.transform(s)
                                states_tensor.append((r, h))
                        else:
                            states_tensor = states

                        for i_intrinsic in range(len(rewards)):
                            if type(states_tensor[i_intrinsic]) is tuple:
                                embeddings.append(self.intrinsic_reward_alg.get_embeddings((states_tensor[i_intrinsic][0].unsqueeze(0),
                                                                   states_tensor[i_intrinsic][1].unsqueeze(0))).to(states_tensor[0][0].device))
                            else:
                                embeddings.append(self.intrinsic_reward_alg.get_embeddings(states_tensor[i_intrinsic].unsqueeze(0).to(states_tensor[0].device)))
                    self.update_memory(states, actions, rewards, local_maps, imitation_learning, embeddings)


                    #self.update_memory(states, actions, rewards, local_maps, imitation_learning)
                    print(len(self.memory))
            #lsy:start
            #设置一个控制是否要更新ped_memory的变量，模仿学习要更新memory吗？
            if update_ped_memory:
                #if isinstance(info, ReachGoal) or isinstance(info, Collision):
                #只需要预测行人的轨迹就行，所以用ob即可，不需要使用last_state
                self.update_ped_memory(ped_states)

            #lsy:end

            #计算这个回合的的总价值函数，和论文中公式（1）相同，之前的reward只是单步的价值
            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            #计算改回合每个时刻的状态价值函数
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            #计算平均价值函数
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / k
        collision_rate = collision / k
        collisionobs_rate = collisionobs / k
        assert success + collision + timeout + collisionobs == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, collisionobs rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f},'
                     ' average return: {:.4f}'. format(phase.upper(), extra_info, success_rate, collision_rate, collisionobs_rate,
                                                       avg_nav_time, average(cumulative_rewards),
                                                       average(average_returns)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times + collisionobs_times)
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         discomfort / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Collision with obstacle cases: ' + ' '.join([str(x) for x in collisionobs_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        #最后返回的是这k个回合的总价值函数的平均值和平均价值函数的平均值
        self.statistics = success_rate, collision_rate, collisionobs_rate, avg_nav_time, average(cumulative_rewards), average(average_returns)

        return self.statistics

    #因为并没有修改V_ex所以要构造两种不同形状的memory 
    def update_memory(self, states, actions, rewards, local_maps, imitation_learning=False, embeddings = None):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]
            action = actions[i]

            local_map = local_maps[i].to(self.device)
            next_local_map = local_maps[i+1].to(self.device)

            # VALUE UPDATE
            #在模仿学习的探索过程中为真
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i+1])
                #这里的时间用的相对时间
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            #在强化学习中执行这个分支，只有当下一步达到目标位置时，才有奖励，奖励就为当步的奖励，其他奖励都为0
            else:
                next_state = states[i+1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)
            action = torch.Tensor([actions[i]]).to(self.device)

            if self.target_policy.name == 'ModelPredictiveRL':
                if self.use_RE3:
                    self.memory.push((state[0], state[1], local_map, action, value, reward, next_state[0], next_state[1],
                                      next_local_map, embeddings[i]))
                else:
                    self.memory.push((state[0], state[1], local_map, action, value, reward, next_state[0], next_state[1],
                                      next_local_map))
                #memory里面记载的是每一步的当前的状态的机器人状态，所有行人的状态以及价值函数单步的奖励以及下一个状态的机器人状态，所有行人的状态，这里很像概率转移公式
                #self.memory.push((state[0], state[1], local_map, action, value, reward, next_state[0], next_state[1], next_local_map))
            else:
                self.memory.push((state, local_map, action, value, reward, next_state, next_local_map))

    #lsy:start
    #用这个函数来更新行人轨迹预测需要的数据，需要包含四个信息的一维向量（时间，行人id,x,y）
    def update_ped_memory(self, ped_states):
        if self.ped_memory is None :
            raise ValueError('Ped_Memory  is not set!')
            #这里是将一次探索的轨迹一次性放入ped_memory里面
            #self.ped_memory.epoch_push(ped_states)
        #for i in range(len(ped_states)):
        if self.ped_memory_batch.first :
            self.ped_memory.push(ped_states)
            self.ped_memory_batch.create(self.ped_memory)
            '''
            self.ped_memory_batch = TrajectoryDataset(
                self.ped_memory,
                obs_len=8,
                pred_len=12,
                skip=1,norm_lap_matr=True)
            '''
        else:
            self.ped_memory_batch.push(ped_states)

    #lsy:end

    #lsy:start
    #step计算得到的ob里面的元素是类，这里用列表ped_data来存放ob里面类的元素，并加入时间和行人id这两个特征
    def ob_ped_data(self, time_step, ob):
        ped_data = []
        ped_id = 1
        for human_state in ob:
            single_ped = []
            single_ped.append(time_step)
            single_ped.append(ped_id)
            single_ped.append(human_state.px)
            single_ped.append(human_state.py)
            ped_data.append(single_ped)
            ped_id = ped_id + 1
        return ped_data
    #lsy:end


    def log(self, tag_prefix, global_step):
        sr, cr, cor, time, reward, avg_return = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/collisionobs_rate', cor, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)

def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
