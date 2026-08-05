import torch as th
from components.action_selectors import REGISTRY as action_selector_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from modules.models.belief_network import BayesianBeliefNetwork

from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class ADVMAC(BasicMAC):
    """
    将信念（belief）作为输入提供给了策略网络
    ADVMAC is a multi-agent controller that conditions each agent's policy
    on a belief about the types of other agents. It uses a BayesianBeliefNetwork
    to infer these beliefs from observations.
    """

    def __init__(self, scheme, groups, args, vae_controller=None):
        # Initialize BasicMAC components without building the agent yet
        self.n_agents = args.n_agents
        self.args = args
        self.vae_controller = vae_controller
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_selector_REGISTRY[args.action_selector](args)
        self.hidden_states = None

        # 初始化信念网络
        obs_shape = scheme["obs"]["vshape"]
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
            
        # Fix: Add agent ID dimension to belief network input if enabled
        if getattr(args, "obs_agent_id", False):
            obs_shape = (obs_shape[0] + self.n_agents,)

        self.belief_network = BayesianBeliefNetwork(
            obs_dim=obs_shape,
            # The following arguments should be added to the config file
            hidden_sizes=getattr(args, "belief_hidden_sizes", [64, 64]),
            num_agents=self.n_agents,
            use_rnn=getattr(args, "belief_use_rnn", True),
            args=args,
        )
        self.belief_hidden_states = None
        self.belief_network_optimizer = th.optim.Adam(
            self.belief_network.parameters(), lr=args.lr_belief
        )

        # 调整actor输入形状以包含信念
        original_input_shape = self._get_input_shape(scheme)
        belief_dim = self.n_agents
        new_input_shape = original_input_shape + belief_dim

        #  构建具有新输入形状的代理
        self._build_agents(new_input_shape)

        # 构建具有相同输入形状的对抗性代理
        self._build_adv_agents(new_input_shape)

        # 动作空间的属性 ---
        self.num_cooperative_actions = getattr(args, "n_coop_actions", 0)
        self.num_adversarial_actions = getattr(args, "n_adv_actions", 0)
        self.hard_belief_threshold = getattr(args, "hard_belief_thres", 0.5)

    def select_actions(self, ep_batch, t_ep, t_env, bs_idx=None, test_mode=False, mac_indices=None, **kwargs):
        
        # This is a legacy method for single-timestep action selection during rollouts.
        # It slices the batch at t_ep and calls the new batch-oriented forward method.
        
        batch_at_t = ep_batch[:, t_ep]

        source_avail_actions = batch_at_t["avail_actions"]
        avail_actions = source_avail_actions.squeeze(1)


        beliefs, policy_logits, predicted_z = self.forward(ep_batch, t_ep, test_mode=test_mode, mac_indices=mac_indices, bs_idx=bs_idx)
        
       
        adv_policy_logits = self.forward_adv(batch_at_t, test_mode=test_mode, predicted_z=predicted_z, mac_indices=mac_indices)
        masked_policy_logits = policy_logits.clone()
        masked_policy_logits[avail_actions == 0] = -1e10
        chosen_actions = self.action_selector.select_action(
            masked_policy_logits, avail_actions, t_env, test_mode=test_mode
        )
        
        masked_adv_policy_logits = adv_policy_logits.clone()
        masked_adv_policy_logits[avail_actions == 0] = -1e10
        chosen_adv_actions = self.action_selector.select_action(
            masked_adv_policy_logits, avail_actions, t_env, test_mode=test_mode
        )
        
        return chosen_actions, chosen_adv_actions, beliefs

    def forward(self, ep_batch, t=None, test_mode=False, mac_indices=None, bs_idx=None):
        """
        Generates action logits for a batch of episode data, processing all timesteps at once.
        This method integrates belief inference, VAE feature extraction, and action masking.
        """
        batch_size = ep_batch.batch_size
        t_max = ep_batch.max_seq_length
        
        # --- Step 1: Belief Inference ---
        # Prepare observations with IDs if needed
        obs = ep_batch["obs"]
        input_obs_dim = self.args.obs_shape
        
        if getattr(self.args, "obs_agent_id", False):
            # Generate one-hot agent IDs
            # Shape: [batch_size, max_seq_length, n_agents, n_agents]
            agent_ids = th.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).unsqueeze(0).expand(batch_size, t_max, -1, -1)
            # Concatenate to obs
            obs = th.cat([obs, agent_ids], dim=-1)
            input_obs_dim += self.n_agents

        obs_flat = obs.reshape(-1, input_obs_dim)
        
        belief_h_in = None
        if getattr(self.args, "belief_use_rnn", True):
            if mac_indices is None: # During training
                belief_h_in = self.belief_hidden_states.reshape(-1, self.belief_hidden_states.shape[-1]) if self.belief_hidden_states is not None else None
                beliefs_flat, belief_h_out_flat = self.belief_network(obs_flat, belief_h_in)
            else: # During rollout
                # Select observations for the current timestep and active environments
                obs_t = obs[:, t]
                if bs_idx is not None:
                    obs_t = obs_t[bs_idx]
                obs_flat_t = obs_t.reshape(-1, input_obs_dim)

                if self.belief_hidden_states is not None:
                    agent_indices = th.tensor([i * self.n_agents + j for i in mac_indices for j in range(self.n_agents)]).to(self.args.device)
                    if bs_idx is not None:
                        # Map mac_indices to the subset defined by bs_idx
                        # This requires careful indexing, assuming mac_indices are indices within the bs_idx subset
                        # Let's rebuild agent_indices based on the size of obs_flat_t
                        num_active_envs = len(bs_idx)
                        agent_indices = th.arange(num_active_envs * self.n_agents).to(self.args.device)
                        
                    belief_h_in = self.belief_hidden_states.index_select(0, agent_indices)
                
                beliefs_flat, belief_h_out_flat = self.belief_network(obs_flat_t, belief_h_in)
        else:
             beliefs_flat, belief_h_out_flat = self.belief_network(obs_flat, belief_h_in)

        
        if getattr(self.args, "belief_use_rnn", True):
            if mac_indices is None: # During training, update the whole hidden state
                self.belief_hidden_states = belief_h_out_flat.reshape(batch_size, self.n_agents, -1)
            else: # During rollout, update slices
                if self.belief_hidden_states is None: # Lazy init
                    full_batch_size = self.args.batch_size_run
                    self.belief_hidden_states = belief_h_out_flat.new_zeros(full_batch_size * self.n_agents, belief_h_out_flat.shape[-1])
                agent_indices = th.tensor([i * self.n_agents + j for i in mac_indices for j in range(self.n_agents)]).to(self.args.device)
                self.belief_hidden_states.index_copy_(0, agent_indices, belief_h_out_flat)

        if mac_indices is None:  # Training
            beliefs = beliefs_flat.view(batch_size, t_max, self.n_agents, self.n_agents)
        else:  # Rollout
            beliefs = beliefs_flat.view(batch_size, 1, self.n_agents, self.n_agents)

        # --- Step 2: VAE Feature Extraction ---
        predicted_z = None
        if self.args.use_dynamics and self.args.use_z_inputs:
            if mac_indices is None: # Training
                # VAE expects [B*T, N, Dims], we have [B, T, N, Dims]
                # This part is complex to batch over time due to t-1 dependencies.
                # For now, we keep it as a loop, which is still a bottleneck but smaller than the main one.
                # A full refactor would require changing the VAE controller itself.
                predicted_z = th.zeros((batch_size, t_max, self.n_agents, self.vae_controller.state_embedding_shape)).to(self.args.device)
                for t_loop in range(t_max):
                    actions_onehot = ep_batch["actions_onehot"][:, t_loop-1] if t_loop > 0 else th.zeros_like(ep_batch["actions_onehot"][:, 0])
                    rewards = ep_batch["reward"][:, t_loop-1] if t_loop > 0 else th.zeros_like(ep_batch["reward"][:, 0])
                    
                    rewards = (rewards - self.vae_controller.rew_ms.mean) / th.sqrt(self.vae_controller.rew_ms.var)
                    rewards = rewards.view(-1, 1, 1).repeat(1, self.n_agents, 1)

                    obs = (ep_batch["obs"][:, t_loop] - self.vae_controller.obs_ms.mean) / (th.sqrt(self.vae_controller.obs_ms.var) + 1e-8)
                    inputs = obs.view(batch_size, self.n_agents, -1)
                    beliefs_for_vae = beliefs[:, t_loop].view(batch_size, self.n_agents, -1)
                    
                    for agent_id in range(self.n_agents):
                        agent_obs_input = inputs[:, agent_id, :]
                        agent_belief_input = beliefs_for_vae[:, agent_id, :]
                        agent_predicted_z = self.vae_controller.forward(agent_obs_input, agent_id, belief=agent_belief_input, test_mode=True)
                        predicted_z[:, t_loop, agent_id, :] = agent_predicted_z
            else: # Rollout
                current_t = t # The timestep passed from select_actions
                predicted_z = th.zeros((batch_size, 1, self.n_agents, self.vae_controller.state_embedding_shape)).to(self.args.device)
                
                actions_onehot = ep_batch["actions_onehot"][:, current_t-1] if current_t > 0 else th.zeros_like(ep_batch["actions_onehot"][:, 0])
                rewards = ep_batch["reward"][:, current_t-1] if current_t > 0 else th.zeros_like(ep_batch["reward"][:, 0])
                
                rewards = (rewards - self.vae_controller.rew_ms.mean) / th.sqrt(self.vae_controller.rew_ms.var)
                rewards = rewards.view(-1, 1, 1).repeat(1, self.n_agents, 1)

                obs = (ep_batch["obs"][:, current_t] - self.vae_controller.obs_ms.mean) / (th.sqrt(self.vae_controller.obs_ms.var) + 1e-8)
                inputs = obs.view(batch_size, self.n_agents, -1)
                
                # Beliefs has shape (bs, 1, n_agents, n_agents) in rollout, so index with 0
                beliefs_for_vae = beliefs[:, 0].view(batch_size, self.n_agents, -1)
                
                for agent_id in range(self.n_agents):
                    agent_obs_input = inputs[:, agent_id, :]
                    agent_belief_input = beliefs_for_vae[:, agent_id, :]
                    agent_predicted_z = self.vae_controller.forward(agent_obs_input, agent_id, belief=agent_belief_input, test_mode=True)
                    predicted_z[:, 0, agent_id, :] = agent_predicted_z
            
            if self.args.use_detach:
                predicted_z = predicted_z.detach()

        # --- Step 3: Policy Calculation ---
        agent_inputs = self._build_inputs(ep_batch, t, mac_indices=mac_indices, predicted_z=predicted_z)
        new_agent_inputs = th.cat([agent_inputs, beliefs.detach()], dim=-1)

        # The logic for training (mac_indices is None) and rollout is different
        if mac_indices is not None: # Rollout - processing a single timestep
            bs = ep_batch.batch_size
            
            # The hidden state is a single tensor of shape [bs * n_agents, hidden_dim].
            # We need to select the slices corresponding to the active environments.
            # First, create a tensor of all agent indices for the active environments.
            agent_indices = th.tensor(
                [i * self.n_agents + j for i in mac_indices for j in range(self.n_agents)],
                device=ep_batch.device
            )
            hidden_state_slice = self.hidden_states.index_select(0, agent_indices)
            
            # The rnn_ns agent expects a 2D hidden state: [bs * n_agents, hidden_dim]
            # which is already the shape of hidden_state_slice.
            correct_hidden_state = hidden_state_slice

            # The rnn_ns agent expects inputs of shape [bs * n_agents, input_dim]
            policy_logits_flat, new_hidden_state_flat = self.agent(
                new_agent_inputs.view(bs * self.n_agents, -1),
                correct_hidden_state
            )
            
            # The new hidden state must be copied back to the correct indices in the full hidden_states tensor.
            self.hidden_states.index_copy_(0, agent_indices, new_hidden_state_flat)
            
            policy_logits = policy_logits_flat.view(bs, self.n_agents, -1)
            
            return beliefs.squeeze(1), policy_logits, predicted_z.squeeze(1) if predicted_z is not None else None

        else: # Training - processing a sequence
            bs = ep_batch.batch_size
            t_max = ep_batch.max_seq_length
            
            # --- This block is replaced to handle recurrent state correctly over the sequence ---
            h_states = self.hidden_states.reshape(-1, self.args.hidden_dim) if self.hidden_states is not None else None
            policy_logits_list = []
            for t_step in range(t_max):
                current_inputs = new_agent_inputs[:, t_step]  # Shape: [bs, n_agents, input_dim]
                current_inputs_flat = current_inputs.reshape(-1, current_inputs.shape[-1])  # Shape: [bs * n_agents, input_dim]
                
                logits_flat, h_states = self.agent(current_inputs_flat, h_states)
                
                logits = logits_flat.view(bs, self.n_agents, -1)
                policy_logits_list.append(logits.unsqueeze(1))
            
            policy_logits = th.cat(policy_logits_list, dim=1)
            # --- End of corrected recurrent handling ---

            # --- Step 4: Belief-Conditioned Action Masking ---
            if self.num_adversarial_actions > 0:
                hard_beliefs = (beliefs > self.hard_belief_threshold).float()
                self_mask = (1 - th.eye(self.n_agents, device=ep_batch.device)).expand(bs, t_max, -1, -1)
                believes_adversary_exists = (hard_beliefs * self_mask).sum(dim=3) > 0
                
                belief_action_mask = th.ones_like(policy_logits)
                adversarial_action_start_index = self.num_cooperative_actions
                belief_action_mask[~believes_adversary_exists, adversarial_action_start_index:] = 0

                final_avail_actions = ep_batch["avail_actions"] * belief_action_mask
            else:
                final_avail_actions = ep_batch["avail_actions"]

            if self.agent_output_type == "pi_logits":
                if getattr(self.args, "mask_before_softmax", True):
                    policy_logits[final_avail_actions == 0] = -1e10
            
            return beliefs, policy_logits, predicted_z


    def forward_adv(self, ep_batch, test_mode=False, predicted_z=None, mac_indices=None):
        # This method is only called during rollouts (mac_indices is not None)
        bs = ep_batch.batch_size
        t_max = ep_batch.max_seq_length # Should be 1 for rollouts

        # 1. Build inputs for the adversary agent, ensuring predicted_z is included.
        # The ep_batch is already sliced for t, so we pass t=0 to _build_inputs.
        # We need to unsqueeze predicted_z to match the time dimension of the batch slice.
        if predicted_z is not None:
            predicted_z = predicted_z.unsqueeze(1)
        agent_inputs = self._build_inputs(ep_batch, t=0, mac_indices=mac_indices, predicted_z=predicted_z)
        
        # 2. Create idealized beliefs (each agent believes it is the adversary)
        idealized_beliefs = th.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).unsqueeze(0).expand(bs, t_max, -1, -1)
        adversarial_agent_inputs = th.cat([agent_inputs, idealized_beliefs], dim=-1)

        # 3. Correctly handle hidden states for the adv_agent (RNNAgent)
        # The shape is [bs, n_agents, hidden_dim]
        adv_hidden_state_slice = self.adv_hidden_states[mac_indices]
        correct_hidden_state = adv_hidden_state_slice.reshape(-1, adv_hidden_state_slice.shape[-1])

        adversarial_policy_logits_flat, new_adv_hidden_state_flat = self.adv_agent(
            adversarial_agent_inputs.view(bs * self.n_agents, -1),
            correct_hidden_state
        )

        # 4. Correctly update the hidden states
        self.adv_hidden_states[mac_indices] = new_adv_hidden_state_flat.view(bs, self.n_agents, -1)

        adversarial_policy_logits = adversarial_policy_logits_flat.view(bs, self.n_agents, -1)
        return adversarial_policy_logits

    def init_hidden(self, batch_size):
        # This is called at the start of a rollout, so batch_size is batch_size_run
        
        # For rnn_ns (self.agent), hidden_states are managed per-agent and concatenated.
        # The shape should be (batch_size * n_agents, hidden_dim)
        if self.args.agent == "rnn_ns":
            self.hidden_states = self.agent.init_hidden().repeat(batch_size * self.n_agents, 1)
        else: # Default for RNNAgent
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).repeat(batch_size, self.n_agents, 1)

        self.belief_hidden_states = None # Will be lazily initialized
        
        # For rnn_agent (self.adv_agent), hidden state is per-agent.
        if self.args.adv_agent == "adv_rnn_agent": # Assuming this corresponds to RNNAgent
             self.adv_hidden_states = self.adv_agent.init_hidden().unsqueeze(0).repeat(batch_size, self.n_agents, 1)
        else: # Fallback or for other types like rnn_ns
            self.adv_hidden_states = self.adv_agent.init_hidden().repeat(batch_size * self.n_agents, 1)

    def cuda(self):
        self.agent.to(self.args.device)
        if self.belief_network is not None:
            self.belief_network.to(self.args.device)
        if hasattr(self, 'adv_agent'):
            self.adv_agent.to(self.args.device)

    def save_models(self, path):
        super().save_models(path)
        th.save(self.adv_agent.state_dict(), f"{path}/adv_agent.th")
        th.save(self.belief_network.state_dict(), f"{path}/belief_network.th")
        th.save(
            self.belief_network_optimizer.state_dict(),
            f"{path}/belief_network_opt.th",
        )

    def load_models(self, path):
        super().load_models(path)
        self.adv_agent.load_state_dict(
            th.load(
                f"{path}/adv_agent.th",
                map_location=lambda storage, loc: storage,
            )
        )
        self.belief_network.load_state_dict(
            th.load(
                f"{path}/belief_network.th",
                map_location=lambda storage, loc: storage,
            )
        )
        self.belief_network_optimizer.load_state_dict(
            th.load(
                f"{path}/belief_network_opt.th",
                map_location=lambda storage, loc: storage,
            )
        )
    
    def _build_adv_agents(self, input_shape):
        adv_agent_type = getattr(self.args, "adv_agent", self.args.agent)
        self.adv_agent = agent_REGISTRY[adv_agent_type](input_shape, self.args)

    def _build_inputs(self, batch, t, mac_indices=None, predicted_z=None):
        # Overwrites the parent method to handle batched sequences
        bs = batch.batch_size
        max_t = batch.max_seq_length
        is_rollout = mac_indices is not None

        inputs = []
        # Obs and Dynamics (z)
        obs_data = batch["obs"]
        if is_rollout:
            current_obs = obs_data[:, t:t + 1]
        else:
            current_obs = obs_data
        
        if self.args.use_dynamics and self.args.use_z_inputs and predicted_z is not None:
            inputs.append(th.cat([current_obs, predicted_z], dim=-1))
        else:
            inputs.append(current_obs)

        # Last Action
        if self.args.obs_last_action:
            if getattr(batch, "actions_onehot", None) is None:
                # If actions_onehot is not present, create zeros. This happens at t=0.
                actions_onehot = th.zeros(bs, max_t, self.n_agents, self.args.n_actions, device=batch.device)
            else:
                actions_onehot = batch["actions_onehot"]
            
            # For t=0, the last action is zeros. For t>0, it's the action from t-1.
            last_actions_seq = th.cat([th.zeros_like(actions_onehot[:, [0]]), actions_onehot[:, :-1]], dim=1)
            if is_rollout:
                inputs.append(last_actions_seq[:, t:t + 1])
            else:
                inputs.append(last_actions_seq)

        # Agent ID
        if self.args.obs_agent_id:
            agent_ids_seq = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
            if is_rollout:
                inputs.append(agent_ids_seq[:, t:t + 1])
            else:
                inputs.append(agent_ids_seq)
            
        inputs = th.cat(inputs, dim=-1)
        return inputs

    def get_differentiable_beliefs(self, ep_batch):
        """
        为 TTA 设计：重新计算信念，保留计算图，以便 VAE 的梯度能反向传播回信念网络。
        """
        batch_size = ep_batch.batch_size
        t_max = ep_batch.max_seq_length
        obs = ep_batch["obs"]
        input_obs_dim = self.args.obs_shape
        
        if getattr(self.args, "obs_agent_id", False):
            agent_ids = th.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).unsqueeze(0).expand(batch_size, t_max, -1, -1)
            obs = th.cat([obs, agent_ids], dim=-1)
            input_obs_dim += self.n_agents

        obs_flat = obs.reshape(-1, input_obs_dim)
        
        # 通过信念网络进行前向传播，保留梯度
        if getattr(self.args, "belief_use_rnn", True):
            # TTA 通常在 episode 结束后更新，这里为简化计算，可以使用初始全零隐状态
            belief_h_in = th.zeros(
                getattr(self.args, "recurrent_N", 1), 
                obs_flat.size(0), 
                self.belief_network.base.output_dim, 
                device=obs.device
            )
            beliefs_flat, _ = self.belief_network(obs_flat, belief_h_in)
        else:
            beliefs_flat = self.belief_network(obs_flat)

        # 还原形状: [batch_size, t_max, n_agents, n_agents]
        differentiable_beliefs = beliefs_flat.view(batch_size, t_max, self.n_agents, self.n_agents)
        return differentiable_beliefs

    def compute_fgsm_obs_perturbation(self, ep_batch, t, adv_agent_ids, envs_not_terminated,
                                    bs_idx=None, mac_indices=None, epsilon=0.1,
                                    fgsm_loss_type="max_entropy"):
        """
        计算 FGSM 观测扰动。
        
        原理：对目标智能体的观测求梯度，然后沿梯度符号方向施加扰动，
        使得协作策略输出尽可能差的动作。
        
        Args:
            ep_batch: 当前 episode batch（未终止环境的子集）
            t: 当前时间步
            adv_agent_ids: list，每个未终止环境中被攻击的智能体 ID
            envs_not_terminated: list，未终止环境的索引
            bs_idx: 可选，batch 子索引
            mac_indices: 环境索引列表（用于 hidden state 切片）
            epsilon: FGSM 扰动强度
            fgsm_loss_type: 损失函数类型
                - "max_entropy": 最大化策略熵（使策略趋于随机）
                - "min_correct": 最小化当前最优动作的概率
                - "targeted": 引导向特定（最差）动作
        
        Returns:
            perturbed_obs: 扰动后的观测 tensor，形状与 ep_batch["obs"][:, t] 相同
                        只有被攻击智能体的观测被修改
        """
        bs = ep_batch.batch_size
        
        # 1. 取出当前时间步的观测，并对被攻击智能体的观测开启梯度
        obs_t = ep_batch["obs"][:, t].clone().detach()  # [bs, n_agents, obs_dim]
        
        # 为被攻击的智能体创建需要梯度的观测副本
        obs_t.requires_grad_(True)
        
        # 2. 用当前观测做一次前向传播（不更新 hidden state）
        #    我们需要保存并恢复 hidden states，因为这只是为了计算梯度
        saved_hidden_states = self.hidden_states.clone() if self.hidden_states is not None else None
        saved_belief_hidden_states = self.belief_hidden_states.clone() if self.belief_hidden_states is not None else None
        
        # 临时替换 batch 中的 obs
        original_obs = ep_batch["obs"][:, t].clone()
        ep_batch["obs"][:, t] = obs_t
        
        # 前向传播获取 policy logits
        beliefs, policy_logits, predicted_z = self.forward(
            ep_batch, t, test_mode=True, mac_indices=mac_indices, bs_idx=bs_idx
        )
        # policy_logits: [bs, n_agents, n_actions]
        
        # 3. 计算损失函数（目标：使策略变差）
        avail_actions = ep_batch["avail_actions"][:, t]  # [bs, n_agents, n_actions]
        
        # 对 policy logits 做 masked softmax
        masked_logits = policy_logits.clone()
        masked_logits[avail_actions == 0] = -1e10
        pi_probs = th.nn.functional.softmax(masked_logits, dim=-1)
        log_pi = th.nn.functional.log_softmax(masked_logits, dim=-1)
        
        loss = th.tensor(0.0, device=ep_batch.device, requires_grad=True)
        attack_count = 0
        
        for i in range(bs):
            adv_id = adv_agent_ids[i]
            if adv_id < 0:
                continue  # 非对抗环境，跳过
            
            agent_probs = pi_probs[i, adv_id]    # [n_actions]
            agent_log_pi = log_pi[i, adv_id]     # [n_actions]
            agent_avail = avail_actions[i, adv_id]  # [n_actions]
            
            if fgsm_loss_type == "max_entropy":
                # 最大化熵 → 使策略趋于均匀随机 → 降低协作性能
                # Loss = -Entropy = sum(p * log(p))，梯度上升 → 最大化熵
                entropy = -th.sum(agent_probs * agent_log_pi)
                loss = loss - entropy  # 我们要最小化 -entropy，即最大化 entropy
                
            elif fgsm_loss_type == "min_correct":
                # 最小化当前最优动作的概率
                best_action = th.argmax(agent_probs).item()
                loss = loss + agent_log_pi[best_action]  # 最小化 best action 的 log prob
                
            elif fgsm_loss_type == "targeted":
                # 引导策略选择最差动作（概率最低的合法动作）
                valid_probs = agent_probs.clone()
                valid_probs[agent_avail == 0] = float('inf')  # 排除不可用动作
                worst_action = th.argmin(valid_probs).item()
                loss = loss - agent_log_pi[worst_action]  # 最大化 worst action 的概率
            
            attack_count += 1
        
        if attack_count == 0:
            # 没有需要攻击的环境，恢复状态后直接返回原始 obs
            self.hidden_states = saved_hidden_states
            self.belief_hidden_states = saved_belief_hidden_states
            ep_batch["obs"][:, t] = original_obs
            return original_obs
        
        loss = loss / attack_count
        
        # 4. 反向传播，计算 obs 的梯度
        loss.backward()
        
        # 5. 获取梯度并计算 FGSM 扰动
        obs_grad = obs_t.grad.data  # [bs, n_agents, obs_dim]
        
        # 6. 构造扰动后的观测（只扰动被攻击智能体）
        perturbed_obs = original_obs.clone()
        
        for i in range(bs):
            adv_id = adv_agent_ids[i]
            if adv_id < 0:
                continue
            perturbation = epsilon * th.sign(obs_grad[i, adv_id])
            perturbed_obs[i, adv_id] = original_obs[i, adv_id] + perturbation
        
        # 7. 恢复 hidden states（因为前向传播修改了它们）
        self.hidden_states = saved_hidden_states
        self.belief_hidden_states = saved_belief_hidden_states
        
        # 8. 恢复原始 obs（稍后会用 perturbed_obs 覆盖）
        ep_batch["obs"][:, t] = original_obs
        
        return perturbed_obs