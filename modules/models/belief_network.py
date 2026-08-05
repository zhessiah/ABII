import torch as th
import torch.nn as nn
import torch.nn.functional as F

class MLPBase(nn.Module):
    """特征提取基础网络"""
    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()
        
        input_dim = obs_shape[0]  # 获取实际的输入维度
        hidden_sizes = getattr(args, "hidden_sizes", [64, 64])  # 从args获取或使用默认值
        use_feature_normalization = getattr(args, "use_feature_normalization", True)
        
        if use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)
        self.use_feature_normalization = use_feature_normalization
        
        layers = []
        last_dim = input_dim
        for hidden_size in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(last_dim, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            ])
            last_dim = hidden_size
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = last_dim
        
    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)
        return self.mlp(x)

class BeliefProj(nn.Module):
    """信念投影网络"""
    def __init__(self, input_dim, num_agents, init_method="orthogonal", gain=0.01):
        super(BeliefProj, self).__init__()
        self.fc = nn.Linear(input_dim, num_agents)
        # 使用较小的gain初始化
        if init_method == "orthogonal":
            nn.init.orthogonal_(self.fc.weight.data, gain=gain)
        else:
            nn.init.xavier_uniform_(self.fc.weight.data, gain=gain)
        nn.init.constant_(self.fc.bias.data, 0)
        
    def forward(self, x):
        return self.fc(x)

class BayesianBeliefNetwork(nn.Module):
    """贝叶斯信念网络 - 与EIR-MAPPO保持一致"""
    def __init__(self, obs_dim, hidden_sizes, num_agents, use_rnn=True, args=None):
        super(BayesianBeliefNetwork, self).__init__()
        self.num_agents = num_agents
        self.use_rnn = use_rnn
        self.args = args if args is not None else type('Args', (), {})()
        
        # 处理 obs_dim：如果是元组，提取第一个元素；如果是整数，直接使用
        if isinstance(obs_dim, tuple):
            obs_dim = obs_dim[0]
        elif isinstance(obs_dim, list):
            obs_dim = obs_dim[0]
        
        # 特征提取器
        self.base = MLPBase(self.args, [obs_dim])
        base_output_dim = self.base.output_dim
        
        # RNN层(可选)
        if use_rnn:
            self.rnn = nn.GRU(
                input_size=base_output_dim,
                hidden_size=base_output_dim,
                num_layers=getattr(self.args, "recurrent_N", 1),
                batch_first=True
            )
        
        # 信念投影层
        self.belief = BeliefProj(
            base_output_dim, 
            num_agents,
            getattr(self.args, "initialization_method", "orthogonal"),
            getattr(self.args, "gain", 0.01)
        )
        
        self.hard_belief_thres = getattr(self.args, "hard_belief_thres", 0.5)
        
    def forward(self, obs, rnn_states=None, masks=None, belief_option='soft'):
        """
        Args:
            obs: [batch_size, obs_dim] 当前观察
            rnn_states: RNN隐状态(如果使用RNN)
            masks: [batch_size] 重置mask
            belief_option: 'soft' 或 'hard' 信念表示方式
        Returns:
            beliefs: [batch_size, num_agents] 对每个智能体的类型信念
            rnn_states: 更新后的RNN隐状态(如果使用RNN)
        """
        
        if getattr(self.args, "disable_belief", False):
            batch_size = obs.size(0)
            # 输出全 0 的 logits，等价于 Softmax 后的均匀概率 (1/N)，表示无偏好/无信息
            beliefs = th.zeros(batch_size, self.num_agents, device=obs.device)
            
            if self.use_rnn:
                if rnn_states is None:
                    # 保持与下游接口一致的隐藏状态形状
                    rnn_states = th.zeros(
                        getattr(self.args, "recurrent_N", 1), 
                        batch_size, 
                        self.base.output_dim, 
                        device=obs.device
                    )
                elif rnn_states.dim() == 2:
                    rnn_states = rnn_states.unsqueeze(0)
                return beliefs, rnn_states.squeeze(0)
            return beliefs
        
        
        # 1. 特征提取
        features = self.base(obs)
        
        # 2. RNN处理(如果启用)
        if self.use_rnn:
            if rnn_states is None:
                # If no hidden state is provided, initialize a 3D one as expected by nn.GRU
                rnn_states = th.zeros(
                    getattr(self.args, "recurrent_N", 1), # num_layers
                    features.size(0), # batch_size
                    features.size(-1), # hidden_size
                    device=features.device
                )
            else:
                # If a 2D hidden state is provided, unsqueeze it to make it 3D
                if rnn_states.dim() == 2:
                    rnn_states = rnn_states.unsqueeze(0)

            if masks is not None:
                features = features * masks.unsqueeze(-1)
            
            features, rnn_states = self.rnn(features.unsqueeze(1), rnn_states)
            features = features.squeeze(1)
        
        # 3. 生成信念
        beliefs = self.belief(features)
        
        # 4. 硬信念处理(可选)
        if belief_option == 'hard':
            beliefs = th.where(beliefs > self.hard_belief_thres, 1.0, 0.0)
        
        if self.use_rnn:
            # Squeeze the hidden state back to 2D to maintain a consistent interface with the controller
            return beliefs, rnn_states.squeeze(0)
        return beliefs