# 多智能体强化学习训练主程序

import datetime
import os
import pprint
import time
import sys
import threading
import torch as th
from types import SimpleNamespace as SN
import numpy as np
import copy
from utils.coustom_logging import Logger
from utils.timehelper import time_left, time_str

from utils.json_logger import JSONLogger
from utils.txt_logger import TxtLogger
from os.path import dirname, abspath, join

# 导入各个模块的注册表
from learners import REGISTRY as le_REGISTRY          # 学习器注册表
from runners import REGISTRY as r_REGISTRY            # 运行器注册表
from model_learners import REGISTRY as mle_REGISTRY   # 模型学习器注册表
from controllers import REGISTRY as mac_REGISTRY      # 控制器注册表
from components.episode_buffer import ReplayBuffer    # 经验回放缓冲区
from components.transforms import OneHot              # One-hot编码转换器
from controllers.centralized_controller import permutate_state  # 状态置换函数
from components.simhash import HashCount             # 哈希计数器


def run(_run, _config, _log):
    """主运行函数：初始化和启动训练过程
    Args:
        _run: Sacred实验运行实例
        _config: 配置参数字典
        _log: 日志记录器
    """
    # 设定日志文件路径
    log_file_path = os.path.join(dirname(abspath(__file__)), 'train.log')
    # 重定向标准输出和标准错误到日志文件
    sys.stdout = open(log_file_path, 'w', buffering=1, encoding='utf-8')
    sys.stderr = sys.stdout

    # 检查参数合法性
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    # if args.evaluate:
    #     _log.info("Evaluation mode detected. Forcing batch_size_run to 1 to save resources.")
    #     args.batch_size_run = 1
        
    if not args.use_cuda:
        args.device = "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info(f"Running on device: {args.device}")

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]   
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
           dirname( dirname(dirname(abspath(__file__)))), "results", "a2r_smpe", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=5)
            print("Thread joined")

    print("Exiting script")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # Making sure framework really exits
    # os._exit(os.EX_OK)
    
def perform_tta_update(args, learner, episode_batch, step_count):
    """
    TTA (测试期自适应) 无监督更新函数。
    该函数仅利用环境的观测(obs)进行自监督重建，绝对不使用真实的 reward。
    """
    # 1. 确保 Actor 和 Critic 的参数被死死冻结，防止作弊
    for param in learner.mac.agent.parameters():
        param.requires_grad = False
    for param in learner.critic.parameters():
        param.requires_grad = False
        
    # 2. ✅ 调用真正的 TTA 在线微调函数 (更新信念网络 + VAE感知层)
    learner.train_tta(episode_batch, step_count)




# def evaluate_sequential(args, runner):

#     for _ in range(args.evaluate_t_max):
#         runner.run(test_mode=True)

#     if args.save_replay:
#         runner.save_replay()

#     runner.close_env()
 
def evaluate_sequential(args, runner, learner=None, buffer=None):
    target_test_steps = int(getattr(args, "test_total_steps", 320000))
    current_test_steps = 0
    ep_idx = 0

    use_tta = getattr(args, "use_tta", False)

    if use_tta:
        runner.logger.console_logger.info(">>> Running in TTA (Test-Time Adaptation) Mode! <<<")
        if learner is None:
            raise ValueError("TTA mode requires learner to be passed to evaluate_sequential.")
    else:
        runner.logger.console_logger.info(">>> Running in Standard Evaluation Mode (Frozen Weights) <<<")

    runner.logger.console_logger.info(f"Starting evaluation for {target_test_steps} env-steps...")

    while current_test_steps < target_test_steps:
        episode_batch = runner.run(test_mode=True)
        ep_idx += 1

        # ✅ 真实步数：用 filled mask 统计有效 transition 数
        filled = episode_batch["filled"]
        steps_taken = int(filled.sum().item())
        current_test_steps += steps_taken

        if use_tta:
            if episode_batch.device != args.device:
                episode_batch.to(args.device)
            perform_tta_update(args, learner, episode_batch, current_test_steps)

            if ep_idx % 10 == 0:
                print(f"[TTA] episode={ep_idx} | env_steps={current_test_steps}/{target_test_steps}")

    if getattr(args, "save_replay", False):
        runner.save_replay()
    runner.close_env()



def run_sequential(args, logger):
    """顺序训练函数：执行多智能体强化学习的主要训练循环"""
    
    # 初始化环境运行器

    # 初始化JSON日志记录器
    json_logs_path = os.path.join(args.sacred_path)
    json_logger = JSONLogger(json_logs_path, args.unique_token)
    txt_logger_path = os.path.join(json_logs_path, f"{args.unique_token}_returns.txt")
    txt_winRate_logger_path = os.path.join(json_logs_path, f"{args.unique_token}_winRate.txt")
    
    txt_logger = TxtLogger(txt_logger_path)
    txt_winRate_logger = TxtLogger(txt_winRate_logger_path)
    runner = r_REGISTRY[args.runner](args=args, logger=logger, json_logger=json_logger,txt_logger=txt_logger, txt_winRate_logger=txt_winRate_logger)
    

    # 设置环境信息和基本参数
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]      # 智能体数量
    args.n_actions = env_info["n_actions"]    # 动作空间大小
    args.state_shape = env_info["state_shape"] # 状态空间维度
    args.obs_shape = env_info["obs_shape"]     # 观察空间维度
    args.state_dim = env_info["state_shape"]   # 状态维度

    # 定义数据结构模式
    scheme = {
        "state": {"vshape": env_info["state_shape"]},          # 全局状态
        "obs": {"vshape": env_info["obs_shape"], 
                "group": "agents"},                            # 每个智能体的观察
        "actions": {"vshape": (1,), 
                   "group": "agents", 
                   "dtype": th.long},                          # 智能体的动作
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,                                   # 可用动作掩码
        },
        "reward": {"vshape": (1,)},                            # 奖励信号
        "terminated": {"vshape": (1,), 
                      "dtype": th.uint8},                      # 终止标志
        "adv_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "belief": {"vshape": (args.n_agents,), "group": "agents", "dtype": th.float32},
        "adversary_id": {"vshape": (1,), "dtype": th.long, "episode_const": True},
    }
    args.scheme = scheme
    
    # 定义智能体分组
    groups = {"agents": args.n_agents}
    
    # 定义预处理操作：将动作转换为one-hot编码
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    args.groups = groups
    args.buffer = buffer 

    if args.use_dynamics:
        # Use Dynamics Controller
        dynamics_controller = mle_REGISTRY[args.dynamics_controller](scheme, args)
        args.name += f"_{args.dynamics_controller}"
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, dynamics_controller)
    else:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, json_logger)

    if args.use_cuda:
        learner.cuda()

    # --- Load Checkpoint Logic ---
    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            timestep_to_load = max(timesteps)
        else:
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            # evaluate_sequential(args, runner)
            evaluate_sequential(args, runner, learner, buffer) 
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # --- Pretraining Setup ---
    # 获取预训练步数，默认为 0 (不预训练)
    pretrain_steps = getattr(args, "pretrain_steps", 0)
    
    if getattr(args, "disable_belief", False):
        pretrain_steps = 0
        logger.console_logger.info("Ablation mode (disable_belief=True): Pretraining phase skipped.")
        
    if pretrain_steps > 0:
        logger.console_logger.info(f"Pre-training Phase Enabled for {pretrain_steps} steps.")
        logger.console_logger.info("During this phase, only VAE and Belief Networks will be updated.")
    

    runner.run(test_mode=True)
    
    # start training
    episode = 0
    last_test_T = - args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    count = 0

    # 主训练循环
    print("\n=== Starting Main Training Loop ===")
    
    while runner.t_env <= args.t_max:
        t_env = runner.t_env
        
        # 判断是否处于预训练阶段
        is_pretraining = (t_env < pretrain_steps)

        # 运行一个完整的训练回合
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        
        # 追踪训练奖励并记录到JSON
        train_reward = episode_batch["reward"][:, :-1].sum().item()
        json_logger.log_stat("train_reward", train_reward, runner.t_env)
        
        # 打印训练进度和奖励信息
        if t_env % 1000 == 0:  # 每1000步打印一次
            status_tag = "[PRETRAIN]" if is_pretraining else "[RL_TRAIN]"
            print(f"\n{status_tag} Progress at t_env {t_env}: Reward: {train_reward:.2f}")
        
        # 如果使用动态模型（VAE状态建模）
        if args.use_dynamics:
            # 添加批次数据到VAE控制器
            learner.mac.vae_controller.addBatch(episode_batch)
            learner.mac.vae_controller.update_stats(episode_batch)

        # 当经验回放缓冲区中有足够的数据时进行训练
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # 截断批次，只保留有效时间步
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            count += 250

            if args.use_dynamics: 
                
                # 1. 更新 Filter 目标 (始终执行)
                if args.use_w and count % args.target_update_filter == 0: 
                    learner.mac.vae_controller.update_filters_targets()

                # 2. 训练 VAE (始终执行)
                # 即使在预训练阶段，VAE 的无监督/自监督学习也是必须的，以构建稳定的 z_i
                if count % args.agent_vae_update_period == 0 and t_env <= args.stop_time_vae:      
                    # 控制打印频率
                    if t_env % 5000 == 0:
                         print(f"Training VAE & Belief at count: {count}")
                    learner.mac.vae_controller.train_agent_vaes(count, episode_sample)
                
                # 3. 分支逻辑：预训练 vs 正式 RL 训练
                if is_pretraining and not getattr(args, "disable_belief", False):
                    # === 预训练阶段 ===
                    # 仅训练 Belief Network (利用 Adversary ID 的监督信号)
                    # 跳过 RL 更新，防止策略拟合无意义的 z_i 和 b_i
                    if hasattr(learner, "train_belief_only"):
                        learner.train_belief_only(episode_sample, t_env)
                    else:
                        # Fallback if method not found (should be implemented in learner)
                        pass
                        
                    if t_env % 5000 == 0:
                        logger.console_logger.info(f"[Pretraining] Optimizing Belief Network... ({t_env}/{pretrain_steps})")

                else:
                    # === 正式 RL 阶段 ===
                    # 检查是否刚刚结束预训练
                    if t_env - episode_batch.batch_size * episode_batch.max_seq_length < pretrain_steps <= t_env:
                        logger.console_logger.info(f"\n>>> Pretraining Finished! Starting Actor-Critic Updates... <<<\n")

                    if args.use_intrinsic:
                        new_rewards = learner.mac.vae_controller.add_intrinsic_rewards(episode_sample, t_env,logger=logger)
                    else:
                        new_rewards = None
                    
                    # 执行完整的 RL 训练 (Actor + Critic + Belief + w_critic)
                    learner.train(episode_sample, runner.t_env, episode, new_rewards)         

            else: 
                # 不使用 Dynamics 的情况 (Legacy逻辑)
                # 如果不使用 Dynamics 也不应该进入预训练循环，或者直接跳过
                if not is_pretraining:
                    learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            
            for _ in range(n_test_runs):
                test_batch = runner.run(test_mode=True)
                # 追踪并记录测试奖励到JSON
                test_reward = test_batch["reward"][:, :-1].sum().item()
                json_logger.log_stat("test_reward", test_reward, runner.t_env)

                


        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_models_path, "models", args.unique_token, str(runner.t_env)
            )
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            json_logger.log_stat("episode", episode, runner.t_env) # JSON LOG
            json_logger.save()
            logger.print_recent_stats()
            last_log_T = runner.t_env
    
    
    
    
    txt_logger.close() 
    txt_winRate_logger.close()
    
    # 保存JSON日志
    json_logger.save()
    runner.close_env()
    logger.console_logger.info("Finished Training")



def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
