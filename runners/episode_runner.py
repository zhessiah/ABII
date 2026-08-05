from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th

# EpisodeRunner是环境交互的核心组件
# 它负责收集训练数据和执行智能体动作

class EpisodeRunner:
    # Softmax 函数就是将“表现得分”转换成“选择概率”的完美工具
    def _softmax(self, x, axis=None):
        """Numerically stable softmax."""
        x = x - np.max(x, axis=axis, keepdims=True)
        y = np.exp(x)
        return y / np.sum(y, axis=axis, keepdims=True)

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        # 在对抗者选择逻辑中被用来判断是否处于 victim_interval（受害者间隔）之内
        self.episode_count = 0
        # This will be initialized properly in setup, once n_agents is known
        self.adapt_adv_probs = None

    def setup(self, scheme, groups, preprocess, mac):
        scheme.update({
            "adv_actions": scheme["actions"].copy(),
            "belief": {
                "vshape": (self.args.n_agents,),
                "group": "agents",
                "dtype": th.float32,
            },
            # adversary_id 是一个回合级别的属性
            "adversary_id": {
                "vshape": (1,),
                "dtype": th.long,
                "episode_const": True,
            },
        })
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

        # --- EIR-MAPPO: Initialize adversary attributes that need n_agents ---
        if getattr(self.args, "random_adversary", False):
            self.adapt_adv_probs = np.zeros(self.args.n_agents)

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        # 选择对抗智能体
        agent_adversary_id = -1  # 默认值为-1，表示没有对抗者
        is_adversary_episode = False

        # 只有当随机对抗者功能启用且不在测试模式下时才执行选择
        if getattr(self.args, "random_adversary", False) and not test_mode:
            # 决定是否在当前episode中包含对抗者，基于间隔参数
            victim_interval = getattr(self.args, "victim_interval", 1)
            if self.episode_count % victim_interval == 0:
                if np.random.rand() < getattr(self.args, "adv_prob", 0.5):
                    is_adversary_episode = True
            else:
                is_adversary_episode = True 
            if is_adversary_episode:
                # 如果当前episode是对抗者episode，现在我们选择*哪个*智能体是对抗者
                if getattr(self.args, "adapt_adversary", False):
                    # 自适应选择：过去表现较差的智能体更有可能被选择
                    probs = self._softmax(-1 * self.adapt_adv_probs)
                    agent_adversary_id = np.random.choice(self.args.n_agents, p=probs)
                else:
                    agent_adversary_id = np.random.choice(self.args.n_agents)

        # 将选定的对抗者传递给环境。
        if hasattr(self.env, "set_adversary"):
            self.env.set_adversary(agent_adversary_id)
        # 选择对抗智能体逻辑结束

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            
            # 在batch中存储对抗者ID
            if self.t == 0:
                # 在episode开始时存储一次
                pre_transition_data["adversary_id"] = [(agent_adversary_id,)]


            self.batch.update(pre_transition_data, ts=self.t)

            # 将迄今为止的所有经验传递给智能体
            # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # 获取action, adv_actions, belief
            actions, adv_actions, belief = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, is_adversary_episode=is_adversary_episode, agent_adversary_id=agent_adversary_id)
            
            input_actions = actions.clone()
            if is_adversary_episode:
                # Assuming actions and adv_actions are of shape (1, n_agents, ...)
                # and agent_adversary_id is a valid index.
                input_actions[0, agent_adversary_id] = adv_actions[0, agent_adversary_id]

            reward, terminated, env_info = self.env.step(input_actions[0])
            episode_return += reward

            post_transition_data = {
                # 记录真正作用于环境的动作（若被对抗者接管则为对抗者动作）
                "actions": input_actions,
                "adv_actions": adv_actions,
                "belief": belief,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, adv_actions, _ = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, is_adversary_episode=is_adversary_episode, agent_adversary_id=agent_adversary_id)
        self.batch.update({"actions": actions, "adv_actions": adv_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.episode_count += 1 # Increment episode counter

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
