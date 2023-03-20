import sys
import logging
import argparse
import os
import shutil
import importlib.util
import torch
import gym
import copy
import git
import re
from tensorboardX import SummaryWriter
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import VNRLTrainer, MPRLTrainer
from crowd_nav.utils.memory import ReplayMemory, ReplayMemory_ped, ped_ReplayMemory, TrajectoryDataset
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.random_encoder import RE3

def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def main(args):
    set_random_seeds(args.randomseed)
    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if key == 'y' and not args.resume:
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
    if make_new_dir:
        os.makedirs(args.output_dir)
        #这里是将args.config中的文件拷贝到args.output_dir这个文件夹中的
        shutil.copy(args.config, os.path.join(args.output_dir, 'config.py'))

        # # insert the arguments from command line to the config file
        # with open(os.path.join(args.output_dir, 'config.py'), 'r') as fo:
        #     config_text = fo.read()
        # search_pairs = {r"gcn.X_dim = \d*": "gcn.X_dim = {}".format(args.X_dim),
        #                 r"gcn.num_layer = \d": "gcn.num_layer = {}".format(args.layers),
        #                 r"gcn.similarity_function = '\w*'": "gcn.similarity_function = '{}'".format(args.sim_func),
        #                 r"gcn.layerwise_graph = \w*": "gcn.layerwise_graph = {}".format(args.layerwise_graph),
        #                 r"gcn.skip_connection = \w*": "gcn.skip_connection = {}".format(args.skip_connection)}
        #
        # for find, replace in search_pairs.items():
        #     config_text = re.sub(find, replace, config_text)
        #
        # with open(os.path.join(args.output_dir, 'config.py'), 'w') as fo:
        #     fo.write(config_text)
    #这里这些文件是怎么来的
    args.config = os.path.join(args.output_dir, 'config.py')
    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

    #将一个模块加载到另一个模块中的方法
    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    #repo = git.Repo(search_parent_directories=True)
    #logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
    logging.info('Current config content is :{}'.format(config))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    writer = SummaryWriter(log_dir=args.output_dir)

    # configure policy
    #这里config文件是怎么生成的，内容是怎么放进去的
    policy_config = config.PolicyConfig()
    #config.py文件里面的policy_config.name是固定的为model_predictive_rl,将执行init函数将所有成员变量初始化为None
    policy = policy_factory[policy_config.name]()
    #policy = policy_factory[args.policy]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    #用policy_config类对policy中的参数进行配置，也就是对ModelPredictiveRL这个类中的参数进行配置
    #这里主要是配置策略中的model,model中包换价值估计器和行人状态预测器，这两者主要通过RGL来构造
    policy.configure(policy_config)
    policy.set_device(device)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    env = gym.make('CrowdSim-v0')
    #这里有对人的数量和策略的初始化
    env.configure(env_config)
    #在这里对机器人进行初始化
    robot = Robot(env_config, 'robot')
    robot.time_step = env.time_step
    #在这里将机器人放进环境中，这个机器人还没有探索策略，即将这个robot传递给env中的self.robot
    env.set_robot(robot)

    policy.set_time_step(env.time_step)

    # read training parameters
    train_config = config.TrainConfig(args.debug)
    rl_learning_rate = train_config.train.rl_learning_rate
    train_batches = train_config.train.train_batches
    train_episodes = train_config.train.train_episodes
    sample_episodes = train_config.train.sample_episodes
    target_update_interval = train_config.train.target_update_interval
    evaluation_interval = train_config.train.evaluation_interval
    capacity = train_config.train.capacity
    epsilon_start = train_config.train.epsilon_start
    epsilon_end = train_config.train.epsilon_end
    epsilon_decay = train_config.train.epsilon_decay
    checkpoint_interval = train_config.train.checkpoint_interval
    # configure trainer and explorer
    #回放10000次的经验 memory继承的是dataset这个类
    #当存入数据量大于存储能力capacity后，这个新的item将会循环覆盖之前的memory中的数据
    memory = ReplayMemory(capacity)
    #lsy:start
    #由于训练行人预测模型和价值估计模型需要的训练数据不一样，所以构造一个新的ped_memory来存放行人预测模型
    ped_memory = ReplayMemory_ped(capacity)
    cur_memory = ReplayMemory_ped(capacity)
    ped_memory_batch = TrajectoryDataset(
                obs_len=3,
                pred_len=2,
                skip=1,norm_lap_matr=True)
    #lsy:end
    #这里得到的是价值估计函数的模型
    model = policy.get_model()
    #batch_size大小是100
    batch_size = train_config.trainer.batch_size
    #优化方法是Adam
    optimizer = train_config.trainer.optimizer

    intrinsic_reward_alg = RE3(policy_config, 7, 3, device, 128, beta=train_config.train.beta,
                               beta_schedule=train_config.train.schedule,
                               rho=train_config.train.rho, k_nn=train_config.train.knn)

    if policy_config.name == 'model_predictive_rl':
        #主要是对类MPRLTrainer的成员变量进行初始化
        trainer = MPRLTrainer(use_RE3 = args.use_RE3, value_estimator=model, state_predictor=policy.state_predictor, map_predictor=policy.map_predictor, memory=memory, ped_memory=ped_memory, ped_memory_batch=ped_memory_batch, device=device, policy=policy, writer=writer, batch_size=batch_size, optimizer_str=optimizer, human_num=env.human_num,
                              reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                              freeze_state_predictor=train_config.train.freeze_state_predictor,
                              detach_state_predictor=train_config.train.detach_state_predictor,
                              share_graph_model=policy_config.model_predictive_rl.share_graph_model,
                              intrinsic_reward = intrinsic_reward_alg,
                              )
    else:
        trainer = VNRLTrainer(model, memory, device, policy, batch_size, optimizer, writer)
    #explorer用环境env和robot初始化，explorer的功能主要是执行探索任务，然后将一个回合结束后的状态奖励等保存在memory中
    explorer = Explorer(args.use_RE3, env, robot, device, writer, memory, ped_memory, cur_memory, ped_memory_batch=ped_memory_batch,
                        gamma=policy.gamma, target_policy=policy, intrinsic_reward = intrinsic_reward_alg)


    # imitation learning
    #args.resume代表的含义应该是是否开始进行强化学习
    metrics = {'train_loss':[],  'val_loss':[]}
    
    if args.resume:
        if not os.path.exists(rl_weight_file):
            logging.error('RL weights does not exist')
        policy.load_model(rl_weight_file)
        #model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
        logging.info('Load reinforcement learning trained weights. Resume training')
    elif os.path.exists(il_weight_file):
        #加载本来的模仿学习模型
        policy.load_model(il_weight_file)
        #model.load_state_dict(torch.load(il_weight_file), False)
        logging.info('Load imitation learning trained weights.')
        #LSY:START
        il_episodes = train_config.imitation_learning.il_episodes
        il_policy = train_config.imitation_learning.il_policy
        il_epochs = train_config.imitation_learning.il_epochs
        il_learning_rate = train_config.imitation_learning.il_learning_rate
        #用学习率设置Adam价值函数优化器
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.imitation_learning.safety_space
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        #机器人的探索方式设置为ORCA
        robot.set_policy(il_policy)
        #LSY:END
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, update_ped_memory=True, imitation_learning=True)
        logging.info('Update memory.')
    else:
        il_episodes = train_config.imitation_learning.il_episodes
        il_policy = train_config.imitation_learning.il_policy
        il_epochs = train_config.imitation_learning.il_epochs
        il_learning_rate = train_config.imitation_learning.il_learning_rate
        #用学习率设置Adam价值函数优化器 
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.imitation_learning.safety_space
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        #机器人的探索方式设置为ORCA
        robot.set_policy(il_policy)
        #用ORCA的方式进行探索
        #il_episodes
        #用orca的成功率有0.8,这里完全是用ORCA
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, update_ped_memory=True, imitation_learning=True)
        #用trainer进行训练，模仿学习50次。这里用memory中的数据作为监督样本对价值估计模型和状态预测模型进行训练，状态预测模型的步长是行人人数
        #il_epochs
        trainer.ped_optimize_epoch(num_epochs=il_epochs, obs_seq_len=3, pred_seq_len=2, metrics=metrics)
        trainer.optimize_epoch(il_epochs)
        policy.save_model(il_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    #对模型进行更新，将训练好的模型放入trainer里面的target model
    trainer.update_target_model(model)
    
    # reinforcement imitation_learning          
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_learning_rate(rl_learning_rate)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, update_ped_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    best_val_reward = -1
    best_val_model = None
    # evaluate the model after imitation learning
    #第0次的时候会执行，这个val模式是什么意思，也不会更新memory
    '''
    if episode % evaluation_interval == 0:
        logging.info('Evaluate the model instantly after imitation learning on the validation cases')
        #env.case_size['val']的具体大小就是100,
        #env.case_size['val']
        explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
        explorer.log('val', episode // evaluation_interval)

        if args.test_after_every_eval:
            explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)
            explorer.log('test', episode // evaluation_interval)
    '''

    episode = 0
    #train_episodes的具体数值是10000会训练10000次
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            #以4000为分界点
            #这里episilon从0开始增加，到第4000个episode的时候，epsilon变成0.5，然后epsilon在episode为4000之后都为0.1
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        #在model_predictive_rl中的predict函数里面会用到epsilon
        robot.policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        #这里和模仿学习中的train是一样的
        #在这里的探索过程中，每次的探索结果会更新到memory里面，如果超过了memory的容量，那么就会覆盖掉memory
        #sample_episodes的大小为1，即完成一个回合的探索，价值函数模型就优化一次，每次训练模型的时候memory里面只新加入一个强化学习的结果
        #探索过程中机器人采用新的policy来预测下一步的动作，在model_predictive里面
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, update_ped_memory=True, episode=episode)
        explorer.log('train', episode)
        #这里train_batches大小是100
        trainer.optimize_batch(train_batches, episode)
        #lsy:start
        #用metric保存损失参数信息
        trainer.ped_optimize_batch(episode, num_batches=train_batches, obs_seq_len=3, pred_seq_len=2, metrics=metrics)
        #lsy:end
        episode += 1

        #每1000次才更新一次模型，optimize_batch里面会用到target_model
        if episode % target_update_interval == 0:
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            _, _, _, _, reward, _ = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
            explorer.log('val', episode // evaluation_interval)

            if episode % checkpoint_interval == 0 and reward > best_val_reward:
                best_val_reward = reward
                best_val_model = copy.deepcopy(policy.get_state_dict())
        # test after every evaluation to check how the generalization performance evolves
            if args.test_after_every_eval:
                explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)
                explorer.log('test', episode // evaluation_interval)

        if episode != 0 and episode % checkpoint_interval == 0:
            current_checkpoint = episode // checkpoint_interval - 1
            save_every_checkpoint_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(current_checkpoint) + '.pth'
            policy.save_model(save_every_checkpoint_rl_weight_file)

    # # test with the best val model
    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
        logging.info('Save the best val model with the reward: {}'.format(best_val_reward))
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('--config', type=str, default='configs/icra_benchmark/mp_separate.py')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=17)

    parser.add_argument('--use_RE3', default=False, action='store_true')

    # arguments for GCN
    # parser.add_argument('--X_dim', type=int, default=32)
    # parser.add_argument('--layers', type=int, default=2)
    # parser.add_argument('--sim_func', type=str, default='embedded_gaussian')
    # parser.add_argument('--layerwise_graph', default=False, action='store_true')
    # parser.add_argument('--skip_connection', default=True, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
