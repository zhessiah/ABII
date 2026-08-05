import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry, CentralQCritic, CentralVCritic
from model_learners import REGISTRY as mle_REGISTRY
from components.standarize_stream import RunningMeanStd
import torch.nn.functional as F
import numpy as np

class ActorCriticVLearner:
    def __init__(self, mac, scheme, logger, args, json_logger=None, mle_learner=None):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.json_logger = json_logger

        self.mac = mac
        # Combine parameters from the agent and adversary agent for the optimizer
        self.agent_params = list(self.mac.agent.parameters()) \
                            + list(self.mac.adv_agent.parameters())
        # Note: The belief_network has its own optimizer in the ADVMAC controller
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.critic_w = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic_w = copy.deepcopy(self.critic_w) # Should copy critic_w, not critic

        self.critic_params_w = list(self.critic_w.parameters()) # Should use critic_w's parameters
        self.critic_optimiser_w = Adam(params=self.critic_params_w, lr=args.lr) # Should use critic_params_w

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        
        # === TTA 专属优化器初始化 ===
        tta_params = list(self.mac.belief_network.parameters())
        if hasattr(self.mac, 'vae_controller') and self.mac.vae_controller is not None:
            for agent_id in range(self.n_agents):
                # tta_params += list(self.mac.vae_controller.agent_models[agent_id].parameters())
                tta_params += list(self.mac.vae_controller.filters[agent_id].parameters())
        
        # 建议在 yaml 配置文件中加上 lr_tta (例如 1e-4)
        lr_tta = getattr(args, "lr_tta", 1e-4) 
        self.tta_optimizer = th.optim.Adam(params=tta_params, lr=lr_tta)

        
        device = self.args.device
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            if self.args.use_intrinsic:
                self.rew_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
            else:
                self.rew_ms = RunningMeanStd(shape=(1,), device=device)
        return

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, new_rewards=None):
        # Get the relevant quantities
        if new_rewards is None:
            rewards = batch["reward"][:, :-1]
        else:
            rewards = new_rewards[:, :-1]

        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        # Initialise hidden states
        self.mac.init_hidden(batch.batch_size)

        # Forward pass
        predicted_beliefs, pi, _ = self.mac.forward(batch)

        # Convert logits to log-probabilities and probabilities
        log_pi = F.log_softmax(pi, dim=-1)
        pi_probs = F.softmax(pi, dim=-1)

        # Train Critic
        advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards,
                                                                      critic_mask, t_env)
        advantages = advantages.detach()

        # Calculate policy grad
        actions = actions[:, :-1]
        log_pi_taken = th.gather(log_pi[:, :-1], dim=3, index=actions).squeeze(3)
        pi_taken = th.gather(pi_probs[:, :-1], dim=3, index=actions).squeeze(3)
        
        entropy = -th.sum(pi_probs[:, :-1] * log_pi[:, :-1], dim=-1)
        pg_loss = -((advantages * log_pi_taken + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        
        # 关键修改：根据是否需要训练信念网络决定是否保留计算图
        # 如果 disable_belief=True，则不需要 retain_graph，节省显存
        should_train_belief = not getattr(self.args, "disable_belief", False)
        pg_loss.backward(retain_graph=should_train_belief)

        # --- Belief Network Training (仅在非消融模式下执行) ---
        if should_train_belief:
            # 1. 准备数据 (从原来的 else 块中移回这里)
            adversary_ids = batch["adversary_id"][:, 0].long() # Shape: [B]
            
            # 扩展 labels: [B] -> [B, T, N] -> [B*T*N]
            labels = adversary_ids.unsqueeze(-1).unsqueeze(-1).repeat(1, batch.max_seq_length, self.n_agents)
            labels_flat = labels.view(-1)

            # 扩展 predictions: [B, T, N, N] -> [B*T*N, N]
            # 注意：predicted_beliefs 形状通常是 [B, T, N, N] (每个智能体对其他所有人的预测)
            # 这里需要确保 view 的维度正确。假设 belief_network 输出是 [B*T, N, N] 或类似
            predictions_flat = predicted_beliefs.view(-1, self.n_agents)

            # 准备 Mask
            belief_mask = batch["filled"][:, :].float().repeat(1, 1, self.n_agents).view(-1)

            # 2. 计算 Loss
            valid_indices = belief_mask.nonzero()
            if valid_indices.numel() > 0:
                valid_indices = valid_indices.squeeze()
                valid_preds = predictions_flat[valid_indices]
                valid_labels = labels_flat[valid_indices]

                # 仅当 batch 中存在由 adversary_id 标记的行时计算（过滤掉全 -1 的情况，如果有的话）
                # 通常 adversary_id 在环境中是固定的，合作者 id 可能是 -1 或别的
                # 这里假设 labels 为 -1 表示不需要预测（例如自身对自身，或者无对抗环境）
                active_belief_mask = (valid_labels != -1)
                
                # 双重检查：如果有有效的对抗样本
                if active_belief_mask.any():
                    valid_preds = valid_preds[active_belief_mask]
                    valid_labels = valid_labels[active_belief_mask]

                    if valid_labels.numel() > 0:
                        belief_loss = F.cross_entropy(valid_preds, valid_labels)

                        # Optimize belief network
                        self.mac.belief_network_optimizer.zero_grad()
                        belief_loss.backward()
                        th.nn.utils.clip_grad_norm_(self.mac.belief_network.parameters(), self.args.grad_norm_clip)
                        self.mac.belief_network_optimizer.step()

                        # Logging
                        if t_env - self.log_stats_t >= self.args.learner_log_interval:
                             self.logger.log_stat("belief_loss", belief_loss.item(), t_env)
                             if self.json_logger is not None:
                                 self.json_logger.log_stat("belief_loss", belief_loss.item(), t_env)

        # Agent Step (Update Actor parameters)
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        # Target Update
        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # Logging (Standard stats)
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "v_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)
                if self.json_logger is not None:
                    self.json_logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)
            
            # ... (Logging extra stats) ...
            
            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi_probs[:,:-1].max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            
            if self.json_logger is not None:
                self.json_logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
                self.json_logger.log_stat("pg_loss", pg_loss.item(), t_env)
                self.json_logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
                self.json_logger.log_stat("pi_max", (pi_probs[:,:-1].max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)

            self.log_stats_t = t_env
            
        return pi_taken

  
    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask, t_env):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        
        # =============================================================================
        # 1. 准备数据 (Belief & ID)
        # =============================================================================
        
        # 获取 Belief: [B, T, N, BeliefDim]
        belief_batch = batch["belief"]
        # 维度安全检查与广播
        if belief_batch.dim() == 3: 
            belief_batch = belief_batch.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        belief_batch = belief_batch.to(batch.device)
        
        # 获取 Agent ID (One-hot): [B, T, N, N]
        id_batch = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)

        # =============================================================================
        # 2. 构建 w_critic 的输入 (主观合成状态)
        #    逻辑：w_critic 评估的是 "智能体主观重构的状态" 的价值，从而指导 Filter 学习
        # =============================================================================
        inputs_w = []
        use_w_critic = self.args.use_w and self.args.use_w_critic and t_env % self.args.update_filter_critic == 0
        
        if use_w_critic:
            subjective_states_list = []
            
            for agent_id in range(self.n_agents):
                # a. 获取数据
                obs = batch["obs"][:, :, agent_id] # [B, T, ObsDim]
                belief = batch["belief"][:, :, agent_id] # [B, T, BeliefDim]
                
                # b. 标准化 Obs (与 VAE Controller 中的处理保持一致)
                # 必须使用运行时的均值和方差，因为 Filter 是基于标准化数据训练的
                mu_obs = self.mac.vae_controller.obs_ms.mean
                std_obs = th.sqrt(self.mac.vae_controller.obs_ms.var) + 1e-8
                norm_obs = (obs - mu_obs) / std_obs
                
                
                use_filter = getattr(self.args, "use_am_filter", True)
                
                if use_filter:
                
                    # c. 通过 Filter 预测缺失信息
                    # Filter 输出: [B, T, StateDim - ObsDim]
                    missing_part = self.mac.vae_controller.filters[agent_id](norm_obs, belief)
                    
                    # d. 构建主观完整状态 (Subjective Full State)
                    subjective_state_i = th.cat([norm_obs, missing_part], dim=-1)
                else:
                    # 如果不使用 Filter，则直接用标准化的 Obs 作为主观状态
                    recon_full_state, _, _, _ = self.mac.vae_controller.agent_models[agent_id](norm_obs, belief, test_mode=False)
                    recon_full_state = recon_full_state.detach()
                    obs_dim = norm_obs.shape[-1]
                    recon_missing_part = recon_full_state[:, :, obs_dim:]
                    subjective_state_i = th.cat([norm_obs, recon_missing_part], dim=-1)
                # 添加回 agent 维度: [B, T, 1, StateDim]
                subjective_states_list.append(subjective_state_i.unsqueeze(2))
            
            # e. 堆叠所有智能体的主观状态: [B, T, N, StateDim]
            subjective_states = th.cat(subjective_states_list, dim=2)
            
            # f. 构建 w_critic 输入向量
            # 结构: [主观状态, 信念, ID]
            inputs_w.append(subjective_states)
            inputs_w.append(belief_batch)
            inputs_w.append(id_batch)
            inputs_w = th.cat(inputs_w, dim=-1)

        # =============================================================================
        # 3. 构建 Main Critic 的输入 (真实全局状态)
        # =============================================================================
        inputs = []
        
        # 真实状态: [B, T, N, StateDim]
        input_batch = batch["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        
        inputs.append(input_batch)  # State
        inputs.append(belief_batch) # Belief (已修复缺失)
        inputs.append(id_batch)     # ID
        inputs = th.cat(inputs, dim=-1)

        # =============================================================================
        # 4. 计算目标值 (Target Values)
        # =============================================================================
        with th.no_grad():
            target_vals = target_critic(inputs)
            target_vals = target_vals.squeeze(3)
            
            if use_w_critic:
                target_vals_w = self.target_critic_w(inputs_w)
                target_vals_w = target_vals_w.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            if use_w_critic:
                target_vals_w = target_vals_w * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Compute n-step target returns
        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        if use_w_critic:
            target_returns_w = self.nstep_returns(rewards, mask, target_vals_w, self.args.q_nstep)

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)
            if use_w_critic:
                target_returns_w = (target_returns_w - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # =============================================================================
        # 5. 计算 Loss 并反向传播
        # =============================================================================
        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "v_taken_mean": [],
        }

        # --- Main Critic Update ---
        v = critic(inputs)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # --- W Critic & Filter Update ---
        if use_w_critic:
            v_w = self.critic_w(inputs_w)[:, :-1].squeeze(3)
            td_error_w = (target_returns_w.detach() - v_w)
            masked_td_error_w = td_error_w * mask
            loss_w = (masked_td_error_w ** 2).sum() / mask.sum()

            self.critic_optimiser_w.zero_grad()
            # 同时清空 Filter 的梯度，因为 loss_w 会反向传播到 Filter
            if getattr(self.args, "use_am_filter", True):
                for agent_id in range(self.n_agents):
                    self.mac.vae_controller.filter_optimizers[agent_id].zero_grad()
            
            loss_w.backward()
            
            # Update w_critic
            th.nn.utils.clip_grad_norm_(self.critic_params_w, self.args.grad_norm_clip)
            self.critic_optimiser_w.step()
            
            use_filter = getattr(self.args, "use_am_filter", True) 
    
            if use_filter:
                # Update Filters
                # 这里利用 w_critic 的梯度来更新 Filter，使 Filter 生成更利于估值的"主观状态"
                for agent_id in range(self.n_agents):
                    filter_params = list(self.mac.vae_controller.filters[agent_id].parameters())
                    th.nn.utils.clip_grad_norm_(filter_params, self.args.grad_norm_clip)
                    
                    # 动态调整学习率 (保持原有逻辑)
                    self.mac.vae_controller.filter_optimizers[agent_id].lr = self.args.lr_w_critic
                    self.mac.vae_controller.filter_optimizers[agent_id].step()
                    self.mac.vae_controller.filter_optimizers[agent_id].lr = self.args.lr_filter
                    
            else:
                pass # 如果不使用 Filter，则不需要更新
        
        # Logging
        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["v_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)
        
        return masked_td_error, running_log


    def train_belief_only(self, batch: EpisodeBatch, t_env: int):
        """
        仅训练信念网络 (用于预训练阶段)
        并不更新 Agent (Actor) 和 Critic
        """
        
        self.mac.init_hidden(batch.batch_size)
        
        #  前向传播
        predicted_beliefs, _, _ = self.mac.forward(batch)
      
        adversary_ids = batch["adversary_id"][:, 0].long() # Shape: [B]

        labels = adversary_ids.unsqueeze(-1).unsqueeze(-1).repeat(1, batch.max_seq_length, self.n_agents)
        labels_flat = labels.view(-1)

        predictions_flat = predicted_beliefs.view(-1, self.n_agents)

        belief_mask = batch["filled"][:, :].float().repeat(1, 1, self.n_agents).view(-1)
        
        valid_indices = belief_mask.nonzero()
        if valid_indices.numel() > 0:
            valid_indices = valid_indices.squeeze()
            valid_preds = predictions_flat[valid_indices]
            valid_labels = labels_flat[valid_indices]

            if (adversary_ids != -1).any():
                active_belief_mask = (valid_labels != -1)
                valid_preds = valid_preds[active_belief_mask]
                valid_labels = valid_labels[active_belief_mask]

                if valid_labels.numel() > 0:
                    belief_loss = F.cross_entropy(valid_preds, valid_labels)

                    self.mac.belief_network_optimizer.zero_grad()
                    belief_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.mac.belief_network.parameters(), self.args.grad_norm_clip)
                    self.mac.belief_network_optimizer.step()
                    
                    if t_env % 1000 == 0:
                        self.logger.log_stat("pretrain_belief_loss", belief_loss.item(), t_env)

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])    # shape: (batch_size, maxseqlen-1, num_agents)
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])    # shape: (batch_size, num_agents)
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t+1]
                else:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        return

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        return

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # Also update the w_critic targets
        for target_param, param in zip(self.target_critic_w.parameters(), self.critic_w.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        return
    
    def cuda(self):
        self.mac.cuda()
        self.critic.to(self.args.device)
        self.target_critic.to(self.args.device)
        # Add the missing models
        self.critic_w.to(self.args.device)
        self.target_critic_w.to(self.args.device)
        return
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        return

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        return
    
    def train_tta(self, batch: EpisodeBatch, t_env: int):
        """
        测试期在线微调：仅使用环境的观测数据进行无监督更新。
        Actor 和 Critic 被完全冻结。
        """
        if not hasattr(self.mac, 'vae_controller') or self.mac.vae_controller is None:
            return

        # 1. 重新计算包含计算图的 Beliefs
        diff_beliefs = self.mac.get_differentiable_beliefs(batch)

        # 2. 计算无监督的 Robust ELBO 损失
        tta_loss = self.mac.vae_controller.compute_tta_loss(batch, diff_beliefs)

        # 3. 梯度反向传播并更新感知网络
        self.tta_optimizer.zero_grad()
        tta_loss.backward()
        
        # 防止梯度爆炸
        tta_params = []
        tta_params += list(self.mac.belief_network.parameters())
        for agent_id in range(self.n_agents):
            # tta_params += list(self.mac.vae_controller.agent_models[agent_id].parameters())
            tta_params += list(self.mac.vae_controller.filters[agent_id].parameters())
            
        th.nn.utils.clip_grad_norm_(tta_params, self.args.grad_norm_clip)
        self.tta_optimizer.step()

        # 打印 TTA 损失日志 (可选)
        if t_env % 10 == 0:
            self.logger.console_logger.info(f"[TTA Adaptation] t_env: {t_env} | TTA Unsupervised Loss: {tta_loss.item():.4f}")
