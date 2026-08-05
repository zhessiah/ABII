import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    """变分编码器类，用于将输入数据编码为潜在空间的分布。

    Args:
        input_shape  obs_dim 拼接 belief_dim
        embedding_shape (int): 潜在空间的维度
        args (argparse.Namespace): 配置参数，包含use_cuda等设置

    Attributes:
        kl (float): 存储KL散度值
    """
    def __init__(self, input_shape, embedding_shape, args):
        super(VariationalEncoder, self).__init__()
        self.input_shape = input_shape
        self.args = args
        self.embedding_shape = embedding_shape

        self.fc1 = nn.Linear(self.input_shape, self.embedding_shape)
        self.fc2 = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.mu = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.logvar = nn.Linear(self.embedding_shape, self.embedding_shape)

        self.N = th.distributions.Normal(0, 1)
        if args.use_cuda:
            self.N.loc = self.N.loc.cuda()          # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        else:
            self.N.loc = self.N.loc          # hack to get sampling on the GPU
            self.N.scale = self.N.scale
        self.kl = 0
        return

    def forward(self, obs, belief, test_mode=False):
        """前向传播函数，将输入编码为潜在空间的分布。

        Args:
            obs (torch.Tensor): 观察数据
            belief (torch.Tensor): 智能体类型的信念信息
            test_mode (bool, optional): 是否为测试模式。在测试模式下直接使用均值而不采样。默认为False。

        Returns:
            tuple: (z, mu, sigma)
                - z (torch.Tensor): 采样得到的潜在变量
                - mu (torch.Tensor): 潜在空间的均值
                - sigma (torch.Tensor): 潜在空间的标准差
        """
        inputs = th.cat([obs, belief], dim=-1)
        x = F.leaky_relu(self.fc1(inputs))
        x = F.leaky_relu(self.fc2(x))
        mu = self.mu(x)
        sigma = self.logvar(x)
        sigma = th.clamp(th.exp(sigma), min=self.args.clip_min, max=self.args.clip_max)

        # Reparameterisation trick
        # Sample from a standard normal distribution and scale by sigma, shift by mu
        epsilon = th.randn_like(sigma)
        z = mu + sigma * epsilon

        return z, mu, sigma


class Decoder(nn.Module):
    """解码器类，用于将潜在空间的变量解码回原始数据空间。

    Args:
        embedding_shape (int): 潜在空间的维度
        output_shape (int): 输出数据的维度
        args (argparse.Namespace): 配置参数，包含use_actions等设置
    """
    def __init__(self, embedding_shape, output_shape, args):
        super(Decoder, self).__init__()
        self.args = args
        self.embedding_shape = embedding_shape
        self.output_shape = output_shape

        self.fc1 = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.fc2 = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.fc3 = nn.Linear(self.embedding_shape, self.output_shape)

        if args.use_actions:
            self.fc4 = nn.Linear(self.embedding_shape, self.args.n_actions * (self.args.n_agents - 1))
        return

    def forward(self, z):
        """前向传播函数，将潜在变量解码为输出数据。

        Args:
            z (torch.Tensor): 潜在空间的变量

        Returns:
            torch.Tensor: 解码后的输出数据
        """
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

    def forward_actions(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        out = self.fc4(x)
        return out


# VAE 的任务是重构 state_dim 全局的完整信息向量。
class VAE(nn.Module):
    """变分自编码器(VAE)的完整实现。

    Args:
        input_shape (int): 输入数据的维度
        embedding_shape (int): 潜在空间的维度
        output_dim (int): 输出数据的维度
        args (argparse.Namespace): 配置参数
    """
    def __init__(self, input_shape, embedding_shape, output_dim, args):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_shape, embedding_shape, args)
        self.decoder = Decoder(embedding_shape, output_dim, args)
        return

    def forward(self, obs, belief, test_mode=False):
        z, mu, sigma = self.encoder(obs=obs, belief=belief, test_mode=test_mode)
        return self.decoder(z), z, mu, sigma


class VariationalEncoder_RNN(nn.Module):
    """基于RNN的变分编码器类，用于处理序列数据并编码为潜在空间的分布。

        Args:
            input_shape (int): 输入维度，等于观察维度(obs_dim)与信念维度(belief_dim)的拼接。               
            embedding_shape (int): 潜在空间的维度
            args (argparse.Namespace): 配置参数
    """
    def __init__(self, input_shape, embedding_shape, args):
        super(VariationalEncoder_RNN, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape

        self.rnn = nn.GRUCell(self.input_shape, self.embedding_shape)
        self.h = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.mu = nn.Linear(self.embedding_shape, self.embedding_shape)
        self.logvar = nn.Linear(self.embedding_shape, self.embedding_shape)

        self.N = th.distributions.Normal(0, 1)
        if args.use_cuda:
            self.N.loc = self.N.loc.cuda()          # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        else:
            self.N.loc = self.N.loc          # hack to get sampling on the GPU
            self.N.scale = self.N.scale
        self.kl = 0
        return

    def forward(self, obs, belief, hidden_state, test_mode=False):
        """前向传播函数，将输入编码为潜在空间的分布。

        Args:
            obs (torch.Tensor): 观察数据
            belief (torch.Tensor): 智能体类型的信念信息
            hidden_state (torch.Tensor): RNN的隐藏状态
            test_mode (bool, optional): 是否为测试模式。默认为False。

        Returns:
            tuple: (z, hidden_state, mu, sigma)
                - z (torch.Tensor): 采样得到的潜在变量
                - hidden_state (torch.Tensor): 更新后的隐藏状态
                - mu (torch.Tensor): 潜在空间的均值
                - sigma (torch.Tensor): 潜在空间的标准差
        """
        x = th.cat([obs, belief], dim=-1)
        hidden_state = self.rnn(x, hidden_state)
        h = F.relu(self.h(hidden_state))

        mu = self.mu(h)
        sigma = th.exp(0.5 * self.logvar(h))
        if test_mode:
            z = mu
        else:
            z = mu + sigma * self.N.sample(mu.shape)  # reparameterisation trick
        self.kl = kl_distance(mu, sigma, th.zeros_like(mu), th.ones_like(sigma))
        return z, hidden_state, mu, sigma


class Variational_Encoder_Decoder_RNN(nn.Module):
    def __init__(self, input_shape, embedding_shape, output_shape, args):
        super(Variational_Encoder_Decoder_RNN, self).__init__()
        self.encoder = VariationalEncoder_RNN(input_shape, embedding_shape, args)
        self.decoder = Decoder(embedding_shape, output_shape, args)
        return

    def forward(self, obs, belief, hidden_state, test_mode=False):
        """前向传播函数。

        Args:
            obs (torch.Tensor): 观察数据
            belief (torch.Tensor): 智能体类型的信念信息
            hidden_state (torch.Tensor): RNN的隐藏状态
            test_mode (bool, optional): 是否为测试模式。默认为False。

        Returns:
            tuple: (decoder_output, hidden_state, z, mu, sigma)
                - decoder_output (torch.Tensor): 解码器输出
                - hidden_state (torch.Tensor): 更新后的隐藏状态
                - z (torch.Tensor): 采样得到的潜在变量
                - mu (torch.Tensor): 潜在空间的均值
                - sigma (torch.Tensor): 潜在空间的标准差
        """
        z, hidden_state, mu, sigma = self.encoder(obs=obs, belief=belief, hidden_state=hidden_state, test_mode=test_mode)
        return self.decoder(z), hidden_state, z, mu, sigma


class Variational_Encoder_Decoder(nn.Module):
    def __init__(self, input_shape, embedding_shape, output_shape, args):
        super(Variational_Encoder_Decoder, self).__init__()
        self.encoder = VariationalEncoder(input_shape, embedding_shape, args)
        self.decoder = Decoder(embedding_shape, output_shape, args)
        return

    def forward(self, obs, belief, test_mode=False):
        z, mu, sigma = self.encoder(obs=obs, belief=belief, test_mode=test_mode)
        return self.decoder(z), z, mu, sigma


def kl_distance(mu1, sigma1, mu2, sigma2):
    """计算两个完全因子化高斯分布之间的KL散度。

    Args:
        mu1 (torch.Tensor): 第一个分布的均值
        sigma1 (torch.Tensor): 第一个分布的标准差
        mu2 (torch.Tensor): 第二个分布的均值
        sigma2 (torch.Tensor): 第二个分布的标准差

    Returns:
        torch.Tensor: KL散度值
    """
    # Fully Factorized Gaussians
    numerator = (mu1 - mu2)**2 + (sigma1)**2
    denominator = 2 * (sigma2)**2 + 1e-8
    return th.sum(numerator / denominator + th.log(sigma2) - th.log(sigma1) - 1/2)


class Aux(nn.Module):
    """辅助网络模块，用于额外的预测任务。

    Args:
        input_shape (int): 输入数据的维度
        output_shape (int): 输出数据的维度
        args (argparse.Namespace): 配置参数
    """
    def __init__(self, input_shape, output_shape, args):
        super(Aux, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.fc1 = nn.Linear(self.input_shape, self.output_shape)
        return

    def forward(self, inputs):
        out = self.fc1(inputs)
        return out
    
# Filter 负责预测缺失的 state.dim - obs_dim  维信息。
class Filter(nn.Module):
    """过滤器模块，用于特征选择或变换。

    Args:
        input_shape :观测的维度
        embedding_shape (int): 输出特征的维度
        args (argparse.Namespace): 配置参数，包含use_gumbel, use_2layer_filter等设置
    """
    def __init__(self, input_shape, embedding_shape, args):
        super(Filter, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape
        
        self.use_belief = getattr(args, "use_belief_in_filter", True)
        
        if self.use_belief:
            self.input_shape = input_shape  # obs_dim + belief_dim
        else:
            self.input_shape = input_shape - args.n_agents  # obs_dim only
            
        if self.args.use_gumbel:

            self.fc1 = nn.Linear(self.input_shape, 2*self.embedding_shape)

        else:
            self.fc1 = nn.Linear(self.input_shape, self.embedding_shape)
        self.fc2 = nn.Linear(self.embedding_shape, self.embedding_shape)
        return

    def forward(self, obs, belief):
        """前向传播函数，根据配置参数选择不同的特征选择/变换方式。

        Args:
            obs (torch.Tensor): 观测数据，用于生成基于内容的掩码。
            belief (torch.Tensor): 信念值，用于对内容掩码进行加权。

        Returns:
            torch.Tensor: 经过过滤/变换后的特征
        """
        # 步骤 1: 将观测(obs)和信念(belief)在最后一个维度上拼接，作为网络的输入
        
        # 记录原始的批次和序列维度，以便后续恢复
        bs, seq_len = obs.shape[0], obs.shape[1]
        
        if self.use_belief:
            # 将 (batch, seq_len) 两个维度压平，以适配全连接层(nn.Linear)
             x = th.cat((obs, belief), dim=-1)
        else:
             x = obs 
        
       
        x_reshaped = x.reshape(-1, x.shape[-1])

        # --- 根据配置生成内容掩码 (content_mask) ---
        # 方案 A: 使用Gumbel-Softmax进行离散的、硬性的特征选择
        if self.args.use_gumbel:
            gumbel_input = self.fc1(x_reshaped).view(-1, self.embedding_shape, 2)
            gumbel_output = F.gumbel_softmax(gumbel_input, hard=True)
            content_mask_indices =th.argmax(gumbel_output, dim=-1) # 选择被激活的特征索引
            content_mask = content_mask_indices.float()

        # 方案 B: 使用Sigmoid生成连续的、软性的特征掩码
        else:
            # B.1: 使用两层网络
            if self.args.use_2layer_filter:
                h = F.relu(self.fc1(x_reshaped))
                content_mask = F.sigmoid(self.fc2(h))        
            else:
                content_mask = F.sigmoid(self.fc1(x_reshaped))
            
            if self.args.use_clip_weights: 
                content_mask = th.clamp(content_mask, min=self.args.clip_min, max=self.args.clip_max)
        
        # --- 步骤 2: 使用信念(belief)来调整(gate)内容掩码 ---
        final_mask = content_mask
        
        # 在返回前，将掩码的形状从2D恢复为3D，与输入序列的形状保持一致
        final_mask = final_mask.reshape(bs, seq_len, -1)
        
        return final_mask


