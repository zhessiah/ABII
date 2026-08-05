from modules.dynamics import REGISTRY as mle_model_REGISTRY
from modules.dynamics import VAE, kl_distance, Aux, Filter
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from components.simhash import HashCount

from torch.optim import Adam, RMSprop
from torch.distributions import MultivariateNormal
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operator import itemgetter 
import gc
import time


class VAEController:
    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.scheme = scheme
        self.args = args
        self.log_stats_t = -self.args.learner_log_interval - 1
        
        # 对抗智能体相关参数
        self.random_adversary = getattr(args, "random_adversary", False)  # 是否随机选择对抗智能体
        self.agent_adversary = getattr(args, "agent_adversary", 0)  # 指定的对抗智能体ID
        self.adv_prob = getattr(args, "adv_prob", 0.5)  # 对抗智能体出现概率
        self.victim_interval = getattr(args, "victim_interval", 1)  # 对抗间隔
        self.super_adversary = getattr(args, "super_adversary", False)  # 是否为超级对手
        self.adapt_adversary = getattr(args, "adapt_adversary", False)  # 是否自适应选择对抗智能体
        
        # 观察标准化参数
        self.use_obs_normalization = getattr(args, "use_obs_normalization", True)  # 是否使用观察标准化
        
        # VAE参数
        self.beta = getattr(args, "beta", 0.1)  # VAE的beta参数，控制KL散度的权重
        # 对于对抗智能体，使用更大的beta以增强鲁棒性
        self.adv_beta = getattr(args, "adv_beta", 0.3)  # 对抗智能体的beta参数
        
        # 对抗智能体的自适应概率
        self.adapt_adv_probs = np.zeros(self.n_agents)
        # 当前episode是否包含对抗智能体
        self.episode_adversary = False
        # 当前episode计数
        self.episode_count = 0

        # Input shapes
        self.state_dim = args.state_dim
        self.n_actions = self.args.n_actions
        self.obs_dim = scheme["obs"]["vshape"]
        self.agent_input_shape = self.obs_dim   
        self.full_input_shape = self.state_dim
        
        # 设置维度参数
        self.state_dim = args.state_dim  # 状态维度为30
        # 确保基础维度能整除状态维度
        self.base_dim = self.state_dim  # 直接使用状态维度作为基础维度
        # 设置状态嵌入维度等于状态维度
        self.state_embedding_shape = self.state_dim
        
        # 设置信念维度
        self.belief_dim = self.n_agents  # 信念维度等于智能体数量
        
        print("\n=== 维度配置信息 ===")
        print(f"状态维度 (state_dim): {self.state_dim}")
        print(f"基础维度 (base_dim): {self.base_dim}")
        print(f"状态嵌入维度 (state_embedding_shape): {self.state_embedding_shape}")
        print(f"观察维度 (obs_dim): {self.obs_dim}")
        
        # 设置网络维度
        self.critic_input_dim = self.state_dim + self.belief_dim + self.n_agents  
        
        print("\n=== Critic 输入维度配置 ===")
        print(f"状态维度: {self.state_dim}")
        print(f"智能体数量维度: {self.n_agents}")
        print(f"信念维度: {self.belief_dim}")
        print(f"总Critic输入维度: {self.critic_input_dim}")
        
        # 验证维度匹配
        expected_dim = self.state_dim + self.n_agents
        if self.critic_input_dim != expected_dim:
            print(f"\n警告: Critic输入维度 ({self.critic_input_dim}) 与期望维度 ({expected_dim}) 不匹配")
            # 自动调整维度
            self.critic_input_dim = expected_dim
            print(f"已自动调整Critic输入维度为: {self.critic_input_dim}")
            
        # 打印权重维度信息
        print("\n=== 权重维度配置 ===")
        print(f"期望权重形状: [batch_size, max_t, n_agents, {self.state_dim}]")
        print(f"基础维度: {self.base_dim}")
        print(f"确保权重维度与状态维度匹配: {self.state_dim} == {self.base_dim}")
        
        if getattr(self.args, "use_actions", True):
            self.actions_dim = self.n_actions
            self.actions_criterion = nn.CrossEntropyLoss(reduction='sum')
        else:
            self.actions_dim = 0
            
        if getattr(self.args, "use_rewards", True):
            self.rewards_dim = 1
        else:
            self.rewards_dim = 0
       
        # Replay buffer (dataset)
        self.dataset_is_full = False
        self.dataset_size = 50
        self.dataset_count = 0
        self.dataset = [0 for _ in range(self.dataset_size)]
        self.obs_ms = RunningMeanStd(shape=(self.obs_dim,), device=self.args.device)
        self.state_ms = RunningMeanStd(shape=(self.state_dim,), device=self.args.device)
        self.rew_ms = RunningMeanStd(shape=(1,), device=self.args.device)

        # Build Hashers
        self.build_hashers()

        # Build agent models
        self.build_agent_models()

        if getattr(self.args, "use_aux", True):
            # Build Auxiliary models
            self.build_agent_auxiliary_models()

        if getattr(self.args, "use_w", True):
            self.build_filters()
            self.build_filters_targets()

        # =====================================================================
        # ABE-Count-Belief：信念熵分桶计数表（消融实验用）
        #
        # 与 use_belief_intrinsic（ABE 熵奖励）互斥，用于对比验证
        # "主动定向驱动信念收敛" vs "被动统计信念状态多样性" 的差异。
        #
        # 配置项（yaml）：
        #   use_belief_count_intrinsic : bool，是否启用计数奖励，默认 False
        #   belief_count_n_bins        : int，熵分桶数量，默认 20
        #   belief_count_rew_coeff     : float，计数奖励系数，默认 0.01
        #   belief_count_reset_episode : bool，是否每 episode 重置计数表，默认 False
        #                                True  → 只统计 episode 内多样性（短视）
        #                                False → 跨 episode 累积（推荐，奖励随训练衰减）
        # =====================================================================
        if getattr(self.args, "use_belief_count_intrinsic", False):
            self.belief_count_n_bins = getattr(self.args, "belief_count_n_bins", 20)
            # shape: (n_agents, n_bins)
            # 每个 agent 独立维护一张计数表，记录其信念熵落入各分桶的累计次数
            self.belief_count_table = np.zeros(
                (self.n_agents, self.belief_count_n_bins), dtype=np.float64
            )
            # 最大熵 = log(n_agents)，均匀分布时取到，用于归一化熵到 [0, 1]
            self.belief_entropy_max = float(np.log(self.n_agents + 1e-8))
            print(f"\n=== ABE-Count-Belief 初始化 ===")
            print(f"分桶数量 (n_bins): {self.belief_count_n_bins}")
            print(f"最大信念熵 (entropy_max): {self.belief_entropy_max:.4f}")
            print(f"跨Episode累积计数: {not getattr(self.args, 'belief_count_reset_episode', False)}")

    # =========================================================================
    # ABE-Count-Belief 辅助方法
    # =========================================================================
    def reset_belief_count(self):
        """
        重置信念熵分桶计数表。
        
        在 use_belief_count_reset_episode=True 时，由 runner 在每个 episode
        开始前调用，使计数奖励仅反映当前 episode 内的信念状态多样性。
        
        默认不调用（跨 episode 累积），此时奖励随训练进行自然衰减，
        行为更接近原始 SimHash 探索奖励的设计哲学。
        """
        if hasattr(self, "belief_count_table"):
            self.belief_count_table = np.zeros(
                (self.n_agents, self.belief_count_n_bins), dtype=np.float64
            )

    def build_hashers(self):
        """构建哈希计数器"""
        self.hash_obs = [HashCount(self.obs_dim) for _ in range(self.n_agents)]
        self.hash_z = [HashCount(self.state_embedding_shape) for _ in range(self.n_agents)]
        return

    def build_filters(self):
        """构建状态过滤器"""
        filter_input_dim = self.obs_dim + self.actions_dim + self.rewards_dim + self.belief_dim
        self.filters = nn.ModuleList([
            Filter(filter_input_dim, 
                  # 观测以外的状态信息维度
                  self.state_dim - self.obs_dim,               
                  self.args).to(self.args.device)
            for _ in range(self.n_agents)
        ])
        self.filter_params = [list(model.parameters()) for model in self.filters]
        self.filter_optimizers = [RMSprop(params=param, lr=self.args.lr_filter) 
                                for param in self.filter_params]
        return

    def build_filters_targets(self):
        """构建目标过滤器"""
        filter_input_dim = self.obs_dim + self.actions_dim + self.rewards_dim + self.belief_dim
        self.target_filters = nn.ModuleList([
            Filter(filter_input_dim,
                  self.state_dim - self.obs_dim,
                  self.args).to(self.args.device)
            for _ in range(self.n_agents)
        ])
        self.update_filters_targets()

    def update_filters_targets(self):
        """更新目标过滤器参数"""
        for agent_id in range(self.n_agents):
            self.target_filters[agent_id].load_state_dict(self.filters[agent_id].state_dict())

    def build_agent_models(self):
        """构建智能体模型"""
        self.agent_models = nn.ModuleList()
        self.agent_params = []
        self.agent_optimizers = []
        
        vae_input_dim = self.obs_dim + self.belief_dim
        
        for agent_id in range(self.n_agents):
            model = VAE(
                vae_input_dim,
                self.state_embedding_shape,
                self.state_dim,  # 全局信息维度
                self.args
            ).to(self.args.device)
            lr = self.args.lr_agent_model
            
            self.agent_models.append(model)
            self.agent_params.append(list(model.parameters()))
            self.agent_optimizers.append(RMSprop(params=self.agent_params[-1], lr=lr))
        
        return

    def build_agent_auxiliary_models(self):
        """构建智能体辅助模型"""
        input_shape = 2 * self.state_embedding_shape
        output_shape = self.n_actions
        self.aux_models = nn.ModuleList([
            Aux(input_shape, output_shape, self.args).to(self.args.device)
            for _ in range(self.n_agents)
        ])
        self.aux_agent_params = [list(model.parameters()) for model in self.aux_models]
        self.aux_optimizers = [RMSprop(params=param, lr=self.args.lr_agent_model) 
                             for param in self.aux_agent_params]
        self.aux_criterion = nn.CrossEntropyLoss(reduction='sum')
        return

    def init_hidden(self, testing=False):
        """初始化隐藏状态"""
        batch_size = 1 if testing else self.args.batch_size
        self.hidden_states = th.zeros(self.n_agents, batch_size, 
                                    self.state_embedding_shape).to(self.args.device)
        return

    def update_stats(self, batch):
        """更新统计信息"""
        self.obs_ms.update(batch["obs"])
        self.state_ms.update(batch["state"])
        self.rew_ms.update(batch["reward"])
        return

    def addBatch(self, batch: EpisodeBatch):
        """添加批次数据到数据集"""
        self.dataset[self.dataset_count % self.dataset_size] = batch
        self.dataset_count += 1
        if self.dataset_count >= self.dataset_size:
            self.dataset_is_full = True
        if self.dataset_count == self.dataset_size:
            self.dataset_count = 0
        if np.random.rand() > 0.8:
            gc.collect()
        return

    def sampleBatches(self, batch_size):
        """从数据集中采样批次"""
        if self.dataset_is_full:
            idx = list(np.random.randint(0, self.dataset_size, batch_size))
        else:
            idx = list(np.random.randint(0, self.dataset_count, batch_size))
        return itemgetter(*idx)(self.dataset)

    def forward(self, inputs, agent_id, belief=None, test_mode=False):
        """前向传播"""
        if belief is None:
            if inputs.dim() == 3: # 如果输入是三维，则创建一个全零的信念张量
                belief = th.zeros(inputs.shape[0], inputs.shape[1], self.belief_dim, device=self.args.device)
            else:
                belief = th.zeros(inputs.shape[0], self.belief_dim, device=self.args.device)
            
        # The VAE model expects 'obs' and 'belief' as keyword arguments
        _, z, _, _ = self.agent_models[agent_id].forward(obs=inputs, belief=belief, test_mode=test_mode)
        if getattr(self.args, "use_detach", True):
            z = z.detach()
        return z

    def save_models(self, t_env, path=None):
        """保存模型"""
        if path is None:
            path = f"saves/ed_{t_env}.pth"
        th.save(self.agent_models.state_dict(), path)
        return

    def load_models(self, t_env, path=None):
        """加载模型"""
        if path is None:
            path = f"saves/ed_{t_env}.pth"
        self.agent_models.load_state_dict(th.load(path))
        return

    def save_filters(self, t_env, path=None):
        """保存过滤器"""
        if path is None:
            path = f"saves/filters_{t_env}.pth"
        th.save(self.filters.state_dict(), path)
        return

    def load_filters(self, t_env, path=None):
        """加载过滤器"""
        if path is None:
            path = f"saves/filters_{t_env}.pth"
        self.filters.load_state_dict(th.load(path))
        return

    def train_agent_vaes(self, t_env, batch):
        """
        训练智能体的VAE模型 (已修改以集成信念)。
        此函数现在执行以下操作：
        1. 直接从传入的batch中获取观测(observation)和由控制器生成的信念(belief)。
        2. 调用 AM Filters 获取基于信念的观测掩码。
        3. 计算加权观测目标 (Weighted Target)。
        4. VAE 训练：输入(观测+信念)，目标(加权观测)。
        """
        total_loss, recon_loss, kl_loss = 0, 0, 0
        batch_size_sample = batch.batch_size
        
        for agent_id in range(self.n_agents):
            # =================================================================================
            # 步骤 1: 获取数据 (观测 + 信念)
            # =================================================================================
            # [B, T, belief_dim]
            agent_beliefs = batch["belief"][:, :, agent_id].to(self.args.device) 
            # [B, T, obs_dim]
            agent_obs = batch["obs"][:, :, agent_id].to(self.args.device)

            # 标准化处理
            if self.use_obs_normalization:
                agent_obs = (agent_obs - self.obs_ms.mean) / th.sqrt(self.obs_ms.var + 1e-8)

            # =================================================================================
            # 步骤 2: 获取 Filter 预测的缺失信息并构建目标
            # =================================================================================
            # Filter 输出: [B, T, state_dim - obs_dim] (例如 20)
            missing_info_prediction = self.filters[agent_id](agent_obs, agent_beliefs)
            
            # 构建完整的目标: 观测 [10] + 预测的缺失信息 [20] = [30]
            # 使用 cat 进行拼接
            full_state_target = th.cat([agent_obs, missing_info_prediction], dim=-1)
              
            # 我们希望 VAE 学习去表征这种组合信息，但不希望通过 VAE loss 来更新 Filter
            weighted_target = full_state_target.detach()

            # =================================================================================
            # 步骤 3: 准备 VAE 输入并前向传播
            # =================================================================================
            batch_size, timesteps, obs_dim = agent_obs.shape
            agent_obs_reshaped = agent_obs.reshape(-1, obs_dim)
            beliefs_reshaped = agent_beliefs.reshape(-1, self.belief_dim)

            # VAE 前向传播
            recon_x, z, mu, sigma = self.agent_models[agent_id](obs=agent_obs_reshaped, belief=beliefs_reshaped)
            
            # 将输出调整回 [B, T, state_dim] (注意这里是 state_dim = 30)
            recon_x = recon_x.reshape(batch_size, timesteps, self.state_dim)
            
            # =================================================================================
            # 步骤 4: 计算损失
            # =================================================================================
            
            # 修正1：计算重构损失 (Target 是 30 维的合成状态)
            recon_per_instance = F.mse_loss(recon_x, weighted_target, reduction='none').sum(dim=[1, 2]) # Shape: [B]

            
            # 计算 KL 散度
            logvar = th.log(sigma.pow(2))
            mu_reshaped = mu.reshape(batch_size, timesteps, -1)
            logvar_reshaped = logvar.reshape(batch_size, timesteps, -1)
            kl_per_instance = -0.5 * th.sum(1 + logvar_reshaped - mu_reshaped.pow(2) - logvar_reshaped.exp(), dim=[1, 2]) # Shape: [B]

            # 处理对抗性 Beta 和总损失
            adversary_id_in_batch = batch["adversary_id"][:, 0].long()
            is_adversary_mask = (agent_id == adversary_id_in_batch)
            
            beta_values = th.full((batch_size_sample,), self.beta, device=self.args.device)
            beta_values[is_adversary_mask] = self.adv_beta

            loss_per_instance = recon_per_instance + beta_values * kl_per_instance
            loss = loss_per_instance.sum()

            # --- 模型优化 ---
            self.agent_optimizers[agent_id].zero_grad()
            loss.backward()
            self.agent_optimizers[agent_id].step()

            # --- 记录日志 ---
            total_loss += loss.item()
            recon_loss += recon_per_instance.sum().item()
            kl_loss += kl_per_instance.sum().item()

        avg_loss = total_loss / (batch_size_sample * self.n_agents)
        avg_recon = recon_loss / (batch_size_sample * self.n_agents)
        avg_kl = kl_loss / (batch_size_sample * self.n_agents)
        
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            if getattr(self.args, "use_wandb", False):
                self.logger.log_stat(f"vae_loss_avg", avg_loss, t_env)
                self.logger.log_stat(f"vae_recon_loss_avg", avg_recon, t_env)
                self.logger.log_stat(f"vae_kl_loss_avg", avg_kl, t_env)
            self.log_stats_t = t_env

    def add_intrinsic_rewards(self, batch: EpisodeBatch, t_env=None, logger=None):
        """
        计算并返回每个智能体的内在奖励。

        支持三种互斥的内在奖励模式（通过 yaml 配置切换，用于消融实验）：

        模式一（SABER 完整版）：use_belief_intrinsic = True
            r_ABE = -H(b_t)
            主动定向驱动信念熵减，鼓励智能体执行能加速身份甄别的探索动作。

        模式二（ABE-Count-Belief 消融变体）：use_belief_count_intrinsic = True
            r_Count = 1 / sqrt(N(bin(H(b_t))))
            被动统计信念熵的分桶访问频率，鼓励覆盖多样的信念熵区间。
            与模式一的本质区别：无定向驱动，奖励随计数累积自然衰减至零。

        模式三（no_ABE 消融变体）：两者均为 False
            不提供任何与信念相关的内在奖励。

        注意：use_belief_intrinsic 与 use_belief_count_intrinsic 不应同时为 True，
        代码会优先执行 use_belief_intrinsic 分支并跳过计数分支。
        """
        time_dim = batch["obs"].shape[1]
        new_rewards = th.zeros(self.args.batch_size, time_dim, self.n_agents)

        for agent_id in range(self.n_agents):
            intr_rews_agent = th.zeros(self.args.batch_size, time_dim, device=self.args.device)

            # -----------------------------------------------------------------
            # 原有 z/obs SimHash 探索奖励（与信念奖励正交，可叠加）
            # -----------------------------------------------------------------
            if getattr(self.args, "z_rew_coeff", 0.0) > 0 or getattr(self.args, "obs_rew_coeff", 0.0) > 0:
                obs = batch["obs"][:, :, agent_id, :]
                # Normalize obs and states
                mu_obs = self.obs_ms.mean
                std_obs = th.sqrt(self.obs_ms.var) + 1e-8
                obs = (obs - mu_obs) / std_obs
                z_others = self.forward(obs, agent_id)
                z_others = z_others.detach()
                z_others = z_others.view(-1, self.state_embedding_shape)
                self.hash_z[agent_id].inc_hash(z_others)
                z_rewards = self.hash_z[agent_id].predict(z_others)
                z_rewards = th.tensor(z_rewards, device=self.args.device)
                z_rewards = z_rewards.view(self.args.batch_size, time_dim)

                obs = obs.view(-1, self.obs_dim)
                self.hash_obs[agent_id].inc_hash(obs)
                obs_rewards = self.hash_obs[agent_id].predict(obs)    
                obs_rewards = th.tensor(obs_rewards, device=self.args.device)
                obs_rewards = obs_rewards.view(self.args.batch_size, time_dim)            

                intr_rews_agent = self.args.z_rew_coeff * z_rewards + self.args.obs_rew_coeff * obs_rewards

            # -----------------------------------------------------------------
            # 模式一：ABE 信念熵奖励（SABER 完整版）
            # r = -coeff * H(b_t)
            # 智能体通过最大化该奖励，学习执行能降低信念熵的试探动作。
            # -----------------------------------------------------------------
            if getattr(self.args, "use_belief_intrinsic", False):
                # [B, T, belief_dim]
                agent_beliefs = batch["belief"][:, :, agent_id].to(self.args.device)
                
                # 转换为概率分布
                belief_probs = F.softmax(agent_beliefs, dim=-1)
                
                # 计算信息熵 H(b) = -sum(p * log(p))，shape: [B, T]
                belief_entropy = -(belief_probs * th.log(belief_probs + 1e-9)).sum(dim=-1)
                
                if logger is not None and t_env is not None:
                    logger.log_stat(
                        f"train_belief_entropy_agent_{agent_id}",
                        belief_entropy.mean().item(), t_env
                    )
                
                belief_rew_coeff = getattr(self.args, "belief_rew_coeff", 0.01)
                # 减去熵值：熵越高奖励越低，驱动智能体降低信念不确定性
                intr_rews_agent -= belief_rew_coeff * belief_entropy

            # -----------------------------------------------------------------
            # 模式二：ABE-Count-Belief 信念熵计数奖励（消融变体）
            #
            # 核心逻辑：
            #   1. 计算当前时间步的信念熵 H(b_t)
            #   2. 将熵值归一化后映射到 [0, n_bins-1] 的离散分桶
            #   3. 查询该分桶的历史访问计数 N，奖励为 1/sqrt(N)
            #   4. 更新计数表
            #
            # 与模式一的本质差异：
            #   - 模式一：奖励正比于 -H(b_t)，持续驱动信念收敛
            #   - 模式二：奖励随 N 增大迅速衰减至零，失去持续驱动能力
            #     在 DLA 潜伏场景下，高熵分桶被反复访问后，奖励趋近于零，
            #     智能体失去主动试探动力，信念收敛窗口无法缩短。
            # -----------------------------------------------------------------
            elif getattr(self.args, "use_belief_count_intrinsic", False):
                # [B, T, belief_dim]
                agent_beliefs = batch["belief"][:, :, agent_id].to(self.args.device)

                # step 1: 计算信念熵，shape: [B, T]
                belief_probs = F.softmax(agent_beliefs, dim=-1)
                belief_entropy = -(belief_probs * th.log(belief_probs + 1e-9)).sum(dim=-1)

                # step 2: 归一化到 [0, 1]，映射到分桶索引 [0, n_bins-1]
                # entropy_max = log(n_agents)，均匀分布时取到最大值
                entropy_norm = (belief_entropy / self.belief_entropy_max).clamp(0.0, 1.0)
                # shape: [B, T]，整数分桶索引
                bin_indices = (entropy_norm * (self.belief_count_n_bins - 1)).long()
                bin_indices_np = bin_indices.cpu().numpy()

                # step 3 & 4: 遍历 batch 和时间步，更新计数并查询奖励
                # 使用 numpy 操作计数表（避免 GPU 与 CPU 频繁同步）
                count_rewards_np = np.zeros(
                    (self.args.batch_size, time_dim), dtype=np.float32
                )
                for b in range(self.args.batch_size):
                    for t in range(time_dim):
                        bin_idx = int(bin_indices_np[b, t])
                        # 先更新计数（当前访问计入），再计算奖励
                        self.belief_count_table[agent_id, bin_idx] += 1.0
                        count = self.belief_count_table[agent_id, bin_idx]
                        count_rewards_np[b, t] = 1.0 / np.sqrt(count)

                count_rewards = th.tensor(
                    count_rewards_np, dtype=th.float32, device=self.args.device
                )

                belief_count_rew_coeff = getattr(self.args, "belief_count_rew_coeff", 0.01)
                intr_rews_agent += belief_count_rew_coeff * count_rewards

                # 日志：记录计数奖励均值和信念熵均值，便于与模式一对比
                if logger is not None and t_env is not None:
                    logger.log_stat(
                        f"train_belief_count_reward_agent_{agent_id}",
                        count_rewards.mean().item(), t_env
                    )
                    logger.log_stat(
                        f"train_belief_entropy_agent_{agent_id}",
                        belief_entropy.mean().item(), t_env
                    )

            # -----------------------------------------------------------------
            # 合并外在奖励与内在奖励
            # -----------------------------------------------------------------
            new_rewards[:, :, agent_id] = (
                self.args.true_rew_coeff * batch["reward"][:, :].squeeze(-1)
                + intr_rews_agent
            )

        new_rewards = new_rewards.detach().to(self.args.device)
        return new_rewards

    def compute_tta_loss(self, batch, differentiable_beliefs):
        """
        计算测试期自适应的无监督损失 (Robust ELBO)。
        不使用任何真实的 adversary_id 标签。
        """
        tta_total_loss = 0
        batch_size_sample = batch.batch_size 
        actions = batch["actions"][:, :-1] # [B, T-1, N, 1]
        
        for agent_id in range(self.n_agents):
            # 获取带有梯度的信念
            agent_beliefs = differentiable_beliefs[:, :, agent_id] # [B, T, belief_dim]
            agent_obs = batch["obs"][:, :, agent_id].to(self.args.device)

            if self.use_obs_normalization:
                agent_obs = (agent_obs - self.obs_ms.mean) / th.sqrt(self.obs_ms.var + 1e-8)

            # 1. 信任门控过滤器 (保持梯度)
            missing_info_prediction = self.filters[agent_id](agent_obs, agent_beliefs)
            full_state_target = th.cat([agent_obs, missing_info_prediction], dim=-1)
            weighted_target = full_state_target.detach() # 目标不需要传导 VAE 的梯度

            # 2. VAE 前向传播
            batch_size, timesteps, obs_dim = agent_obs.shape
            agent_obs_reshaped = agent_obs.reshape(-1, obs_dim)
            beliefs_reshaped = agent_beliefs.reshape(-1, self.belief_dim)

            recon_x, z, mu, sigma = self.agent_models[agent_id](obs=agent_obs_reshaped, belief=beliefs_reshaped)
            recon_x = recon_x.reshape(batch_size, timesteps, self.state_dim)
            
            # 3. 计算 ELBO (重构损失 + KL 散度)
            recon_per_instance = F.mse_loss(recon_x, weighted_target, reduction='none').sum(dim=[1, 2])
            
            logvar = th.log(sigma.pow(2))
            mu_reshaped = mu.reshape(batch_size, timesteps, -1)
            logvar_reshaped = logvar.reshape(batch_size, timesteps, -1)
            kl_per_instance = -0.5 * th.sum(1 + logvar_reshaped - mu_reshaped.pow(2) - logvar_reshaped.exp(), dim=[1, 2])

            # 在 TTA 阶段，由于不知道谁是真正的敌人，统一使用标准的 beta
            beta_values = th.full((batch_size_sample,), self.beta, device=self.args.device)
            loss_per_instance = recon_per_instance + beta_values * kl_per_instance
            
            tta_total_loss += loss_per_instance.sum()

        return tta_total_loss / (batch_size_sample * self.n_agents)
