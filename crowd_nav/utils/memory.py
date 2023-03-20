from torch.utils.data import Dataset
import numpy as np
import math
import torch
from tqdm import tqdm
import networkx as nx
import bz2
import _pickle as cPickle
import os

#lsy:start
def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)

#输入是位置，相对位置，以及true                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    #时间序列的长度
    seq_len = seq_.shape[2]
    #行人数量
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        #根据行人数量进行遍历
        for h in range(len(step_)): 
            #V矩阵就是将位置矩阵进行转置，第一维为步长，第二维为行人，第三维为x~y
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                #A矩阵的权重就是距离倒数
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
    #返回的是相对位置矩阵和权重矩阵        
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    #在0到traj_len-1之间以均匀步长生成traj_len个数据，即以1为步长生成数据
    t = np.linspace(0, traj_len - 1, traj_len)
    #用二阶函数分别构造x，y随时间变化的关系，res_x/res_y输出的是多项式系数，从高阶到低阶
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    #考察x,y的变化是不是线性的
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
#lsy:end

class ReplayMemory(Dataset):
    #传入数据，或者像下面一样直接在函数里加载数据
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0
        self.data_path = "/home/cbl/RelationalGraphLearning_staticobs+stgcnn/crowd_nav/memory_buffer"
        #self.path_list = os.listdir(self.data_path)
        self.use_RE3 = False

    #push函数是将一个新的item放入memory里面，position记录下一个存放的位置，当存入数据量大于存储能力capacity后，这个新的item将会循环覆盖之前的memory中的数据
    def push(self, item):
        # replace old experience with new experience
        #if len(self.memory) < self.position + 1:
        #    self.memory.append(item)
        #else:
        #    self.memory[self.position] = item
        if self.use_RE3:
            robot_states, human_states, local_map, actions, values, rewards, next_robot_states, next_human_states, next_local_map, embeddings = item
            path = f'{self.data_path}/{self.position:04d}.pbz2'
            with bz2.BZ2File(path, 'w') as fp:
                cPickle.dump(
                    {
                        'robot_states': robot_states,
                        'human_states': human_states,
                        'local_map': local_map,
                        'actions': actions,
                        'values': values,
                        'rewards': rewards,
                        'next_robot_states': next_robot_states,
                        'next_human_states': next_human_states,
                        'next_local_map': next_local_map,
                        'embeddings': embeddings
                    },
                    fp
                )
        else:
            robot_states, human_states, local_map, actions, values, rewards, next_robot_states, next_human_states, next_local_map = item
            path = f'{self.data_path}/{self.position:04d}.pbz2'
            with bz2.BZ2File(path, 'w') as fp:
                cPickle.dump(
                    {
                        'robot_states': robot_states,
                        'human_states': human_states,
                        'local_map': local_map,
                        'actions': actions,
                        'values': values,
                        'rewards': rewards,
                        'next_robot_states': next_robot_states,
                        'next_human_states': next_human_states,
                        'next_local_map': next_local_map
                    },
                    fp
                )
        if len(self.memory) < self.position + 1:
            self.memory.append(self.position)
        else:
            self.memory[self.position] = self.position
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    #返回一条训练数据，并将其转换成tensor
    def __getitem__(self, item):
        #name = self.path_list[item]
        #item_path = os.path.join(self.data_path, name)
        item_path = f'{self.data_path}/{item:04d}.pbz2'
        if self.use_RE3:
            with bz2.BZ2File(item_path, 'rb') as fp:
                data = cPickle.load(fp)
                robot_states = data['robot_states']
                human_states = data['human_states']
                local_map = data['local_map']
                actions = data['actions']
                values = data['values']
                rewards = data['rewards']
                next_robot_states = data['next_robot_states']
                next_human_states = data['next_human_states']
                next_local_map = data['next_local_map']
                embeddings = data['embeddings']
            return (robot_states, human_states, local_map, actions, values, rewards, next_robot_states,
                                                                  next_human_states, next_local_map, embeddings)
        else:
            with bz2.BZ2File(item_path, 'rb') as fp:
                data = cPickle.load(fp)
                robot_states = data['robot_states']
                human_states = data['human_states']
                local_map = data['local_map']
                actions = data['actions']
                values = data['values']
                rewards = data['rewards']
                next_robot_states = data['next_robot_states']
                next_human_states = data['next_human_states']
                next_local_map = data['next_local_map']
            return (robot_states, human_states, local_map, actions, values, rewards, next_robot_states,
                                                                  next_human_states, next_local_map)
        #return self.memory[item]

    #返回这个数据集一共有多少个item
    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()

#lsy:start
class ReplayMemory_ped(Dataset):
    #传入数据，或者像下面一样直接在函数里加载数据
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    #push函数是将一个新的item放入memory里面，position记录下一个存放的位置，当存入数据量大于存储能力capacity后，这个新的item将会循环覆盖之前的memory中的数据
    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    #返回一条训练数据，并将其转换成tensor
    def __getitem__(self, item):
        return self.memory[item]

    #返回这个数据集一共有多少个item
    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        #self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        #用于预测的历史轨迹序列长度和预测得到的新的轨迹序列长度和
        self.seq_len = self.obs_len + self.pred_len
        #这个是定界符
        #self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.min_ped = min_ped
        self.threshold =threshold
        self.seq_start_end=[0]
        self.first = True

    def create(self, ReplayMemory):
        self.ped_memory = ReplayMemory.memory


        #返回指定的文件夹包含的文件或文件夹的名字的列表。train 文件则返回train数据集中的.txt文件
        #all_files = os.listdir(self.data_dir)
        #这里应该是返回上面文件的绝对路径
        #all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        flag = True
        #for path in all_files:
        for path in range(len(self.ped_memory)):
            #这里用的是Tab键来当定界符
            #data = read_file(path, delim)
            data = self.ped_memory[path]
            #数据中的第一列信息应该是时间信息
            data = np.array(data)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                #frame_data读取数据中每一行的数据，用一个二维数组存放训练数据,将数据按第一项从小到大的顺序重新排列
                #但是这里就是一个三维数据了，第一个维度是时间，第二个维度是行人，第三个维度的前两项可以去掉
                frame_data.append(data[frame == data[:, 0], :])
            #20个时间片的数据为一组
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                #将每组数据分别提取至curr_seq_data，20个数据为一组，取了20个时间步长的数据
                curr_seq_data = np.concatenate(
                    #将frame中的数据每seq_len为一组,可以将三维信息变成二维
                    frame_data[idx:idx + self.seq_len], axis=0)

                #去掉重复值，并由小到大排列，记录当前序列中的行人序列
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                #这个参数记录行人个数
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                #定义了一个三维张量，第一个维度为行人数量，第三个维度为每组数据中数据条数
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                #遍历当前采样序列中的每个行人
                for _, ped_id in enumerate(peds_in_curr_seq):
                    #提取当前行人对应的轨迹数据
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    #对输入保留小数点后四位
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    #找到frames中数据为当前行人数据第一项的索引减去idx，即该行人的第一项数据距离采样
                    #这里pad_end保留了这组数据包含的最早的时间信息，curr_ped_seq是按时间顺序排列的
                    #frames.index是查找当前行人在这组数据中最早最晚时间在时间序列中的索引
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    #如果这组行人信息中没有包含足量的长度为seq_len的时间信息，采用continue
                    #这样做的意义就是要找到在seq_len条时间数据里面，刚好时间步长最大值为seq_len这个长度的数据
                    if pad_end - pad_front != self.seq_len:
                        continue
                    else:
                        flag = False
                        self.first = False
                    #将curr_ped_seq从第三项开始转置变成新的行人数据，则第一个维度表示x~y,第二个维度表示该行人的时间步长seq_len    
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative制造相对位置
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    #计算同一个行人移动的相对位置
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    #num_peds_considerd是在这组相邻数据中被考虑的行人数量
                    _idx = num_peds_considered
                    #这俩就是将所有被考虑行人的位置/相对位置分别记录下来
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory记录当前行人在seq_len的时间步长内轨迹是否是线性变化的
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, self.pred_len, self.threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1


                if num_peds_considered > self.min_ped:
                    #这里相加就是向量内部对应位置的元素相加
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
        if flag :
            return
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        #将数据分为用作历史轨迹的那段和用作预测轨迹的那段，但是都是来源于真实数据集
        #这里数据的三个维度分别为行人，x~y, 时间序列
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        #将相对位移分类
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        #np.cumsum用于将数组按行累加
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        #将相邻两次考虑的行人数量打包
        self.seq_start_end = [
            (start, end)
            #将cum_start_idx对象中两个相邻的元素打包成一个组，作为start和end
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        if len(self.seq_start_end) != 1:
            print("Processing Data .....")
        #‘Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
        if len(self.seq_start_end) != 1:
            pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            if len(self.seq_start_end) != 1:
                pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        if len(self.seq_start_end) != 1:
            pbar.close()

    def push(self, item):
        num_peds_in_seq = [self.seq_start_end[len(self.seq_start_end)-1][1]]
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        data = item
        #数据中的第一列信息应该是时间信息
        data = np.array(data)
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        flag = True
        for frame in frames:
            #frame_data读取数据中每一行的数据，用一个二维数组存放训练数据,将数据按第一项从小到大的顺序重新排列
            #但是这里就是一个三维数据了，第一个维度是时间，第二个维度是行人，第三个维度的前两项可以去掉
            frame_data.append(data[frame == data[:, 0], :])
        #20个时间片的数据为一组
        num_sequences = int(
            math.ceil((len(frames) - self.seq_len + 1) / self.skip))

        
        for idx in range(0, num_sequences * self.skip + 1, self.skip):
            #将每组数据分别提取至curr_seq_data，20个数据为一组，取了20个时间步长的数据
            curr_seq_data = np.concatenate(
                #将frame中的数据每seq_len为一组,可以将三维信息变成二维
                frame_data[idx:idx + self.seq_len], axis=0)

            #去掉重复值，并由小到大排列，记录当前序列中的行人序列
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            #这个参数记录行人个数
            self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
            #定义了一个三维张量，第一个维度为行人数量，第三个维度为每组数据中数据条数
            curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                     self.seq_len))
            curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
            curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                       self.seq_len))
            num_peds_considered = 0
            _non_linear_ped = []
            #遍历当前采样序列中的每个行人
            for _, ped_id in enumerate(peds_in_curr_seq):
                #提取当前行人对应的轨迹数据
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                             ped_id, :]
                #对输入保留小数点后四位
                curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                #找到frames中数据为当前行人数据第一项的索引减去idx，即该行人的第一项数据距离采样
                #这里pad_end保留了这组数据包含的最早的时间信息，curr_ped_seq是按时间顺序排列的
                #frames.index是查找当前行人在这组数据中最早最晚时间在时间序列中的索引
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                #如果这组行人信息中没有包含足量的长度为seq_len的时间信息，采用continue
                #这样做的意义就是要找到在seq_len条时间数据里面，刚好时间步长最大值为seq_len这个长度的数据
                if pad_end - pad_front != self.seq_len:
                    continue
                else:
                    flag = False
                #将curr_ped_seq从第三项开始转置变成新的行人数据，则第一个维度表示x~y,第二个维度表示该行人的时间步长seq_len    
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                curr_ped_seq = curr_ped_seq
                # Make coordinates relative制造相对位置
                rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                #计算同一个行人移动的相对位置
                rel_curr_ped_seq[:, 1:] = \
                    curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                #num_peds_considerd是在这组相邻数据中被考虑的行人数量
                _idx = num_peds_considered
                #这俩就是将所有被考虑行人的位置/相对位置分别记录下来
                curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                # Linear vs Non-Linear Trajectory记录当前行人在seq_len的时间步长内轨迹是否是线性变化的
                _non_linear_ped.append(
                    poly_fit(curr_ped_seq, self.pred_len, self.threshold))
                curr_loss_mask[_idx, pad_front:pad_end] = 1
                num_peds_considered += 1

            if num_peds_considered > self.min_ped:
                #这里相加就是向量内部对应位置的元素相加
                non_linear_ped += _non_linear_ped
                num_peds_in_seq.append(num_peds_considered)
                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                seq_list.append(curr_seq[:num_peds_considered])
                seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        if flag :
            return
        self.num_seq = self.num_seq + len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        cur_obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        cur_pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        #将相对位移分类
        cur_obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        cur_pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        cur_loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        cur_non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

        self.obs_traj = torch.cat((self.obs_traj, cur_obs_traj), 0)
        #print(len(self.obs_traj))
        self.pred_traj = torch.cat((self.pred_traj, cur_pred_traj), 0)
        #print(len(self.pred_traj))
        self.obs_traj_rel = torch.cat((self.obs_traj_rel, cur_obs_traj_rel), 0)
        #print(len(self.obs_traj_rel))
        self.pred_traj_rel = torch.cat((self.pred_traj_rel, cur_pred_traj_rel), 0)
        #print(len(self.pred_traj_rel))
        self.loss_mask = torch.cat((self.loss_mask, cur_loss_mask), 0)
        #print(len(self.loss_mask))
        self.non_linear_ped = torch.cat((self.non_linear_ped, cur_non_linear_ped), 0)
        #print(len(self.non_linear_ped))

        #print(num_peds_in_seq)
        cum_start_idx = np.cumsum(num_peds_in_seq).tolist()
        for start, end in zip(cum_start_idx, cum_start_idx[1:]):
             self.seq_start_end.append((start, end))
        '''
        self.seq_start_end.append(
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        )
        '''
        cur_seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        if len(cur_seq_start_end) > 1000:
            print("update Data .....")
        if len(cur_seq_start_end) > 1000:
            pbar = tqdm(total=len(cur_seq_start_end)) 
        for ss in range(len(cur_seq_start_end)):
            if len(cur_seq_start_end) > 1000:
                pbar.update(1)

            start, end = cur_seq_start_end[ss]
            #print('lsy')
            #print(start)

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        if len(cur_seq_start_end) > 1000:
            pbar.close()


    def __len__(self):
        return self.num_seq

    #返回数据分别是真实位置的观测集，真实位置的预测集；相对位置的观测集，相对位置的预测集；轨迹是非线性行人的集合；掩码集合；真实位置观测集的转置，权重矩阵；相对位置观测集的转置，权重矩阵
    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        #这里得到的就是用于训练的数据集
        return out

#这个类的作用应该是处理所有push进去的数据
class ped_ReplayMemory(Dataset):
    
    def __init__(self, capacity, min_ped=1, norm_lap_matr = True):
        #这里capacitiy可以设置为20,那么超出20个数据之后，data里面的数据就会被更新
        self.seq_len = capacity
        #每个时间步长可以更新一次，一次加入5个行人的数据当加入20个时间步长的数据之后就可以训练了
        #这个data要构造成二维向量，第二个维度的特征分别为时间，行人id,px,py
        self.ped_data = list()
        #当time_position到达20之后开始组合训练数据
        self.time_position = 0
        #总体数据条数
        self.position=0
        self.norm_lap_matr = norm_lap_matr
        self.num_seq = 0
        

    #保存一个时间步长中行人轨迹的信息
    def onetime_push(self,item):
        
        #更新时间步长
        #当新加入数据和前一个数据的时间信息不一样的时候，time_position加一
        if len(self.ped_data) == 0:
            self.ped_data.append(item)
            #print(ped_data)
        elif self.ped_data[-1][0] == item[0]:

            self.ped_data.append(item)

        else:
            self.time_position = self.time_position + 1
            if self.time_position == self.seq_len:#seq_len长度的数据已经被准备好了

                #构造这seq_len长度的训练数据结构
                frames_fir = np.array(self.ped_data)
                frames = np.unique(frames_fir[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    #frame_data读取数据中每一行的数据，用一个二维数组存放训练数据,将数据按第一项从小到大的顺序重新排列
                    #但是这里就是一个三维数据了，第一个维度是时间，第二个维度是行人，第三个维度的前两项可以去掉
                    frame_data.append(frames_fir[frame == frames_fir[:, 0], :])
                curr_seq_data = np.concatenate(
                    #将frame中的数据每seq_len为一组,可以将三维信息变成二维
                    frame_data[:self.seq_len], axis=0)

                #去掉重复值，并由小到大排列，记录当前序列中的行人序列
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                #这个参数记录行人个数
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                #定义了一个三维张量，第一个维度为行人数量，第三个维度为每组数据中数据条数
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                #遍历当前采样序列中的每个行人
                for _, ped_id in enumerate(peds_in_curr_seq):
                    #提取当前行人对应的轨迹数据
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    #对输入保留小数点后四位
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    #找到frames中数据为当前行人数据第一项的索引减去idx，即该行人的第一项数据距离采样
                    #这里pad_end保留了这组数据包含的最早的时间信息，curr_ped_seq是按时间顺序排列的
                    #frames.index是查找当前行人在这组数据中最早最晚时间在时间序列中的索引
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    #如果这组行人信息中没有包含足量的长度为seq_len的时间信息，采用continue
                    #这样做的意义就是要找到在seq_len条时间数据里面，刚好时间步长最大值为seq_len这个长度的数据
                    if pad_end - pad_front != self.seq_len:
                        continue
                    #将curr_ped_seq从第三项开始转置变成新的行人数据，则第一个维度表示x~y,第二个维度表示该行人的时间步长seq_len    
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative制造相对位置
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    #计算同一个行人移动的相对位置
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    #num_peds_considerd是在这组相邻数据中被考虑的行人数量
                    _idx = num_peds_considered
                    #这俩就是将所有被考虑行人的位置/相对位置分别记录下来
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    #这个数据的第一项是没有被定义的不能用
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory记录当前行人在seq_len的时间步长内轨迹是否是线性变化的
                    #pred_len和threshold没有定义
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped = np.asarray(_non_linear_ped)

                self.num_seq = len(curr_seq)
                # Convert numpy -> Torch Tensor
                #将数据分为用作历史轨迹的那段和用作预测轨迹的那段，但是都是来源于真实数据集
                #这里数据的三个维度分别为行人，x~y, 时间序列
                self.obs_traj = torch.from_numpy(
                    #因为定义curr_seq的时候第一维是行人总数，但是有一些行人不符合规范，被剔除掉了
                    curr_seq[:num_peds_considered, :, :self.obs_len]).type(torch.float)
                self.pred_traj = torch.from_numpy(
                    curr_seq[:num_peds_considered, :, self.obs_len:]).type(torch.float)
                #将相对位移分类
                self.obs_traj_rel = torch.from_numpy(
                    curr_seq_rel[:num_peds_considered, :, :self.obs_len]).type(torch.float)
                self.pred_traj_rel = torch.from_numpy(
                    curr_seq_rel[:num_peds_considered, :, self.obs_len:]).type(torch.float)
                self.loss_mask = torch.from_numpy(curr_loss_mask[:num_peds_considered]).type(torch.float)
                self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

                cum_start_idx = [0, num_peds_considered]

                #将相邻两次考虑的行人数量打包
                self.seq_start_end = [
                    (start, end)
                    #将cum_start_idx对象中两个相邻的元素打包成一个组，作为start和end
                    for start, end in zip(cum_start_idx, cum_start_idx[1:])
                ]
                #Convert to Graphs 
                self.v_obs = [] 
                self.A_obs = [] 
                self.v_pred = [] 
                self.A_pred = [] 
                print("Processing Data .....")
                #‘Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
                pbar = tqdm(total=len(self.seq_start_end)) 
                #这里只有一项其实可以把for循环去掉的
                for ss in range(len(self.seq_start_end)):
                    pbar.update(1)

                    start, end = self.seq_start_end[ss]

                    v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
                    self.v_obs.append(v_.clone())
                    self.A_obs.append(a_.clone())
                    v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
                    self.v_pred.append(v_.clone())
                    self.A_pred.append(a_.clone())
                pbar.close()

                #删除第一个时间片刻的数据，把新数据添加到最后面
                self.time_position = self.time_position - 1
                #思路就是找到time_data第一项的时间，然后把时间相同的数据都删掉再在time_data最后面添加新的数据
                while len(self.ped_data) != 0:
                    if self.ped_data[0][0] == self.ped_data[0][0]:
                        self.ped_data.remove(ped_data[0])
                    else:
                        self.ped_data.remove(ped_data[0])
                        break
                self.ped_data.append(item)
            else:
                self.ped_data.append(item)
        
        if len(self.ped_data) < self.time_position + 1:
            self.ped_data.append(item)
        else:
            #当ped_data放满之后就得从头开始放了
            self.ped_data[self.time_position] = item
        if self.time_position+1==self.seq_len:#放满20个时间步长的数据之后（这个判断条件不准确）
            self.position = (self.position + 1) % self.seq_len
        

    #保存整个探索中行人轨迹的信息
    def epoch_push(self, ped_epoch_data):
        for item in range(len(ped_epoch_data)):
            onetime_push(ped_epoch_data[item])


    def is_full(self):
        return len(self.ped_data) == self.capacity

    def __len__(self):
        return self.num_seq

    #返回数据分别是真实位置的观测集，真实位置的预测集；相对位置的观测集，相对位置的预测集；轨迹是非线性行人的集合；掩码集合；真实位置观测集的转置，权重矩阵；相对位置观测集的转置，权重矩阵
    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        #这里得到的就是用于训练的数据集
        return out