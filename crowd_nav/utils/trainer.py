import logging
import abc
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .metrics import * 
from crowd_nav.utils.memory import TrajectoryDataset
import numpy

def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

class MPRLTrainer(object):
    def __init__(self, use_RE3, value_estimator, state_predictor, map_predictor, memory, ped_memory, ped_memory_batch, device, policy, writer, batch_size, optimizer_str, human_num,
                 reduce_sp_update_frequency, freeze_state_predictor, detach_state_predictor, share_graph_model, intrinsic_reward = None):
        """
        Train the trainable model of a policy
        """
        self.value_estimator = value_estimator
        self.state_predictor = state_predictor
        self.map_predictor = map_predictor
        self.device = device
        self.writer = writer
        self.target_policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.criterion_map = nn.SmoothL1Loss(reduction='sum').to(device)
        self.memory = memory
        #lsy:start
        #加载行人memory
        self.ped_memory = ped_memory
        self.ped_memory_batch = ped_memory_batch
        #lsy:end
        self.data_loader = None
        self.loader_train = None
        print(self.loader_train)
        #lsy:start
        #加载行人轨迹数据
        self.ped_data_loader = None
        #lsy:end
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.reduce_sp_update_frequency = reduce_sp_update_frequency
        self.state_predictor_update_interval = human_num
        self.freeze_state_predictor = freeze_state_predictor
        self.detach_state_predictor = detach_state_predictor
        self.share_graph_model = share_graph_model
        self.v_optimizer = None
        self.s_optimizer = None

        # for value update
        self.gamma = 0.9
        self.time_step = 10
        self.v_pref = 2

        self.intrinsic_reward_alg = intrinsic_reward
        self.use_RE3 = use_RE3

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
            if self.map_predictor.trainable:
                self.m_optimizer = optim.Adam(self.map_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate)
            if self.map_predictor.trainable:
                self.m_optimizer = optim.SGD(self.map_predictor.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

        if self.state_predictor.trainable and self.map_predictor.trainable:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters()) +
                 list(self.state_predictor.named_parameters()) + list(self.map_predictor.named_parameters())]), 
                  self.optimizer_str))
        else:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters())]), self.optimizer_str))
    
    def optimize_epoch(self, num_epochs):
        #优化回合里面首先需要优化器v_optimizer，在trainer自定义的时候就生成好了
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)

        #难道用同一组数据训练50次？
        for epoch in range(num_epochs):
            epoch_v_loss = 0
            epoch_s_loss = 0
            epoch_m_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))

            update_counter = 0
            for data in self.data_loader:
                #这里的value是执行orca得到的样本，orca作为监督者，它产生的value作为标签
                if self.use_RE3:
                    robot_states, human_states, local_map, actions, values, _, _, next_human_states, next_local_map, embeddings = data
                else:
                    robot_states, human_states, local_map, actions, values, _, _, next_human_states, next_local_map = data

                #local_map = local_map.to(self.device)
                #next_local_map = next_local_map.to(self.device)

                # optimize value estimator
                #v_optimizer和value_estimator是绑定的
                self.v_optimizer.zero_grad()
                #用nn.module这个神经网络做的,这里返回的是一个值神经网络，这个值神经网络是要被训练的，所以用它的输出和监督者输出的标签进行比较计算损失函数
                outputs = self.value_estimator((robot_states, human_states), local_map)    #local_map参与计算值估计
                values = values.to(self.device)
                #这里values是标签再用output来计算损失
                loss = self.criterion(outputs, values)
                #loss = self.criterion(values, values)
                #loss.requires_grad_(True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_estimator.parameters(), 1.)
                #优化器用step方法来对参数进行更新
                self.v_optimizer.step()
                epoch_v_loss += loss.data.item()

                # optimize state predictor
                if self.map_predictor.trainable:
                    update_state_predictor = True
                    if update_counter % self.state_predictor_update_interval != 0:
                        update_state_predictor = False

                    if update_state_predictor:
                        self.m_optimizer.zero_grad()
                        next_local_map_est = self.map_predictor(local_map, actions)
                        loss_map = self.criterion_map(next_local_map_est, next_local_map)
                        loss_map.backward()
                        torch.nn.utils.clip_grad_norm_(self.map_predictor.parameters(), 1.)
                        self.m_optimizer.step()
                        epoch_m_loss += loss_map.data.item()

                    update_counter += 1

            logging.debug('{}-th epoch ends'.format(epoch))
            self.writer.add_scalar('IL/epoch_v_loss', epoch_v_loss / len(self.memory), epoch)
            self.writer.add_scalar('IL/epoch_m_loss', epoch_m_loss / len(self.memory), epoch)
            #logging.info('Average loss in epoch %d: %.2E, %.2E', epoch, epoch_v_loss / len(self.memory),
                         #epoch_s_loss / len(self.memory))
            logging.info('Average loss in epoch %d: %.6f, %.6f', epoch, epoch_v_loss / len(self.memory), epoch_m_loss / len(self.memory))
        return

    #lsy:start
    def ped_optimize_epoch(self, num_epochs, obs_seq_len, pred_seq_len, metrics):
        #优化回合里面首先需要优化器v_optimizer，在trainer自定义的时候就生成好了
        if self.s_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.ped_data_loader is None:
            self.ped_loader_train = DataLoader(
                self.ped_memory_batch,
                batch_size=1, #This is irrelative to the args batch size parameter
                shuffle =True,
                num_workers=0)

        #难道用同一组数据训练50次？
        for epoch in range(num_epochs):
            logging.debug('{}-th epoch starts'.format(epoch))

            update_counter = 0

            self.state_predictor.human_motion_predictor.train()
            loss_batch = 0 
            batch_count = 0
            is_fst_loss = True
            loader_len = len(self.ped_loader_train)
            turn_point =int(loader_len/32)*32 + loader_len%32 -1
            
            # optimize state predictor
            #模仿学习的时候这里为真
            if self.state_predictor.trainable:
                update_state_predictor = True
                for cnt,batch in enumerate(self.ped_loader_train):
                        update_counter += 1

                    #if update_counter % self.state_predictor_update_interval != 0:
                        #update_state_predictor = False

                    #只有训练次数可以整出行人人数才能执行下面的，为什么阿？？？难道是d步长的训练
                    #if update_state_predictor:

                        batch_count+=1

                        #Get data
                        batch = [tensor.cuda() for tensor in batch]
                        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
                         loss_mask,V_obs,A_obs,V_tr,A_tr = batch
                        self.s_optimizer.zero_grad()
                        #Forward
                        #V_obs = batch,seq,node,feat
                        #V_obs_tmp = batch,feat,seq,node
                        V_obs_tmp =V_obs.permute(0,3,1,2)

                        V_pred,_ = self.state_predictor.human_motion_predictor(V_obs_tmp,A_obs.squeeze())
                        
                        V_pred = V_pred.permute(0,2,3,1)

                        V_tr = V_tr.squeeze()
                        A_tr = A_tr.squeeze()
                        V_pred = V_pred.squeeze()

                        if batch_count%32 !=0 and cnt != turn_point :
                            l = graph_loss(V_pred,V_tr)
                            if is_fst_loss :
                                loss = l
                                is_fst_loss = False
                            else:
                                loss += l
                        else:
                            loss = loss/32
                            is_fst_loss = True
                            loss.backward()
                            
                            #if args.clip_grad is not None:
                                #torch.nn.utils.clip_grad_norm_(self.state_predictor.parameters(),args.clip_grad)
                            torch.nn.utils.clip_grad_norm_(self.state_predictor.parameters(), 1.)
                            self.s_optimizer.step()
                            #Metrics
                            loss_batch += loss.item()
                            #print('TRAIN:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)
                    
                metrics['train_loss'].append(loss_batch/batch_count)
                logging.info('Average loss in epoch of statePredictor%d: %.6f', epoch, loss_batch/batch_count)
    #lsy:end

    #lsy:start
    #这个里面需要把轨迹预测模型相关部删除掉
    def optimize_batch(self, num_batches, episode):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=False)
        v_losses = 0
        s_losses = 0
        m_losses = 0
        batch_count = 0
        for data in self.data_loader:

            if self.use_RE3:
                robot_states, human_states, local_map, actions, _, rewards, next_robot_states, next_human_states, \
                                                                                            next_local_map, embeddings = data
            else:
                robot_states, human_states, local_map, actions, _, rewards, next_robot_states, next_human_states, \
                                                                                            next_local_map = data

            if self.use_RE3:
                rewards = rewards + self.intrinsic_reward_alg.compute_intrinsic_reward_batch(embeddings, episode)

            #robot_states, human_states, local_map, actions, _, rewards, next_robot_states, next_human_states, next_local_map = data
            #local_map = local_map.to(self.device)
            #next_local_map = next_local_map.to(self.device)

            # optimize value estimator
            self.v_optimizer.zero_grad()
            outputs = self.value_estimator((robot_states, human_states), local_map)    #local_map参与计算值估计

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            #这里target_model是模仿学习后优化好的模型
            target_values = rewards + gamma_bar * self.target_model((next_robot_states, next_human_states), next_local_map)

            # values = values.to(self.device)
            #难道强化学习和监督学习的loss都可以这样计算吗
            loss = self.criterion(outputs, target_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_estimator.parameters(), 1.)
            self.v_optimizer.step()
            v_losses += loss.data.item()

            # optimize state predictor
            if self.map_predictor.trainable:
                update_state_predictor = True
                if self.freeze_state_predictor:
                    update_state_predictor = False
                elif self.reduce_sp_update_frequency and batch_count % self.state_predictor_update_interval == 0:
                    update_state_predictor = False

                if update_state_predictor:
                    self.m_optimizer.zero_grad()
                    next_local_map_est = self.map_predictor(local_map, actions)
                    loss_map = self.criterion_map(next_local_map_est, next_local_map)
                    loss_map.backward()
                    torch.nn.utils.clip_grad_norm_(self.map_predictor.parameters(), 1.)
                    self.m_optimizer.step()
                    m_losses += loss_map.data.item()

            batch_count += 1
            if batch_count > num_batches:
                break

        average_v_loss = v_losses / num_batches
        average_m_loss = m_losses / num_batches
        #logging.info('Average loss : %.2E, %.2E', average_v_loss, average_s_loss)
        logging.info('Average loss : %.6f, %.6f', average_v_loss, average_m_loss)
        self.writer.add_scalar('RL/average_v_loss', average_v_loss, episode)
        self.writer.add_scalar('RL/average_m_loss', average_m_loss, episode)
        return average_v_loss, average_m_loss
    #lsy:end

    #lsy:start
    #这个函数用于训练行人轨迹预测器
    def ped_optimize_batch(self, episode, num_batches, obs_seq_len, pred_seq_len, metrics):
        if self.s_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.loader_train is None:
            self.loader_train = DataLoader(
                self.ped_memory_batch,
                batch_size=1, #This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=0)

        self.state_predictor.human_motion_predictor.train()
        loss_batch = 0 
        batch_count = 0
        is_fst_loss = True
        loader_len = len(self.loader_train)
        #print(loader_len)
        turn_point =int(loader_len/32)*32 + loader_len%32-1
        if len(self.loader_train)==0:
            return
        if self.state_predictor.trainable:
            update_state_predictor = True
            if update_state_predictor:
                for cnt,batch in enumerate(self.loader_train):
                        #print("1")

                    #if update_counter % self.state_predictor_update_interval != 0:
                        #update_state_predictor = False

                    #只有训练次数可以整出行人人数才能执行下面的，为什么阿？？？难道是d步长的训练
                    #if update_state_predictor:
                        '''
                        if len(self.loader_train) <  episode + num_batches:
                            if cnt + num_batches < len(self.loader_train) :
                                continue
                        else:
                            if cnt < episode  :
                                continue
                        '''
                        batch_count+=1
                        #print(batch_count)

                        #Get data
                        batch = [tensor.cuda() for tensor in batch]
                        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
                         loss_mask,V_obs,A_obs,V_tr,A_tr = batch
                        self.s_optimizer.zero_grad()
                        #Forward
                        #V_obs = batch,seq,node,feat
                        #V_obs_tmp = batch,feat,seq,node
                        V_obs_tmp =V_obs.permute(0,3,1,2)

                        V_pred,_ = self.state_predictor.human_motion_predictor(V_obs_tmp,A_obs.squeeze())
                        
                        V_pred = V_pred.permute(0,2,3,1)

                        V_tr = V_tr.squeeze()
                        A_tr = A_tr.squeeze()
                        V_pred = V_pred.squeeze()

                        if batch_count%32 !=0 and cnt != turn_point :
                            l = graph_loss(V_pred,V_tr)
                            if is_fst_loss :
                                loss = l
                                is_fst_loss = False
                            else:
                                loss += l
                            #print('Average loss of statePredictor: %.2E', loss/batch_count)
                        else:
                            loss = loss/32
                            is_fst_loss = True
                            loss.backward()
                            
                            #if args.clip_grad is not None:
                                #torch.nn.utils.clip_grad_norm_(self.state_predictor.parameters(),args.clip_grad)
                            torch.nn.utils.clip_grad_norm_(self.state_predictor.parameters(), 1.)
                            self.s_optimizer.step()
                            #Metrics
                            loss_batch += loss.item()
                        if batch_count > num_batches:
                            break
                metrics['train_loss'].append(loss_batch/batch_count)
                logging.info('Average loss of statePredictor: %.6f', loss_batch/batch_count)
    '''
    def ped_optimize_batch(self, episode, num_batches, obs_seq_len, pred_seq_len, metrics):
        if self.s_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.loader_train is None:
            self.loader_train = DataLoader(
                self.ped_memory_batch,
                batch_size=1, #This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=0)

        self.state_predictor.human_motion_predictor.train()
        loss_batch = 0 
        batch_count = 0
        is_fst_loss = True
        loader_len = len(self.loader_train)
        turn_point =int(loader_len/128)*128+ loader_len%128-1
        
        if self.state_predictor.trainable:
            update_state_predictor = True
            if update_state_predictor:
                for cnt,batch in enumerate(self.loader_train):

                    #if update_counter % self.state_predictor_update_interval != 0:
                        #update_state_predictor = False

                    #只有训练次数可以整出行人人数才能执行下面的，为什么阿？？？难道是d步长的训练
                    #if update_state_predictor:
                        if len(self.loader_train) <  episode + num_batches:
                            if cnt + num_batches < len(self.loader_train) :
                                continue
                        else:
                            if cnt < episode  :
                                continue

                        batch_count+=1

                        #Get data
                        batch = [tensor.cuda() for tensor in batch]
                        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
                         loss_mask,V_obs,A_obs,V_tr,A_tr = batch
                        self.s_optimizer.zero_grad()
                        #Forward
                        #V_obs = batch,seq,node,feat
                        #V_obs_tmp = batch,feat,seq,node
                        V_obs_tmp =V_obs.permute(0,3,1,2)

                        V_pred,_ = self.state_predictor.human_motion_predictor(V_obs_tmp,A_obs.squeeze())
                        
                        V_pred = V_pred.permute(0,2,3,1)
                        
                        V_tr = V_tr.squeeze()
                        A_tr = A_tr.squeeze()
                        V_pred = V_pred.squeeze()

                        if batch_count%128 !=0 and cnt != turn_point :
                            l = graph_loss(V_pred,V_tr)
                            if is_fst_loss :
                                loss = l
                                is_fst_loss = False
                            else:
                                loss += l
                            #print('Average loss of statePredictor: %.2E', loss/batch_count)
                        else:
                            loss = loss/128
                            is_fst_loss = True
                            loss.backward()
                            
                            #if args.clip_grad is not None:
                                #torch.nn.utils.clip_grad_norm_(self.state_predictor.parameters(),args.clip_grad)


                            self.s_optimizer.step()
                            #Metrics
                            loss_batch += loss.item()

                
                        if batch_count > num_batches:
                            break
                metrics['train_loss'].append(loss_batch/batch_count)
                logging.info('Average loss of statePredictor: %.2E', loss_batch/batch_count)


    #lsy:end
    '''

class VNRLTrainer(object):
    def __init__(self, model, memory, device, policy, batch_size, optimizer_str, writer):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.optimizer = None
        self.writer = writer

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError
        logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
            [name for name, param in self.model.named_parameters()]), self.optimizer_str))

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))
            for data in self.data_loader:
                inputs, values, _, _ = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
            logging.debug('{}-th epoch ends'.format(epoch))
            average_epoch_loss = epoch_loss / len(self.memory)
            self.writer.add_scalar('IL/average_epoch_loss', average_epoch_loss, epoch)
            logging.info('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches, episode=None):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        losses = 0
        batch_count = 0
        for data in self.data_loader:
            inputs, _, rewards, next_states = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model(next_states)

            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()
            batch_count += 1
            if batch_count > num_batches:
                break

        average_loss = losses / num_batches
        logging.info('Average loss : %.2E', average_loss)

        return average_loss


def pad_batch(batch):
    """
    args:
        batch - list of (tensor, label)
    return:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    def sort_states(position):
        # sort the sequences in the decreasing order of length
        sequences = sorted([x[position] for x in batch], reverse=True, key=lambda t: t.size()[0])
        packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
        return torch.nn.utils.rnn.pad_packed_sequence(packed_sequences, batch_first=True)

    states = sort_states(0)
    values = torch.cat([x[1] for x in batch]).unsqueeze(1)
    rewards = torch.cat([x[2] for x in batch]).unsqueeze(1)
    next_states = sort_states(3)

    return states, values, rewards, next_states
