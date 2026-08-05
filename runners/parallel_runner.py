from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process, set_start_method
import numpy as np
import torch as th
import inspect
import time

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger, json_logger=None, txt_logger=None, txt_winRate_logger=None):
        self.args = args
        self.logger = logger
        self.json_logger = json_logger
        self.txt_logger = txt_logger
        self.txt_winRate_logger = txt_winRate_logger
        self.batch_size = self.args.batch_size_run          # parallel envs

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                   for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0
        self.test_steps_total = 0   # 测试期累计"真实环境步数"
        self.test_episode_idx = 0   # 测试期累计 episode 数（按 batch_size_run 累加）

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

        # --- EIR-MAPPO: Adversary attributes ---
        self.total_episodes_run = 0
        # self.adapt_adv_probs will be initialized in setup, once n_agents is known
        self.adapt_adv_probs = None

        # --- FGSM 统计计数器 ---
        self._fgsm_applied_steps = 0   # 本次 run() 内实际触发扰动的时间步数

    def setup(self, scheme, groups, preprocess, mac):
        # Add adversary-related fields to the scheme
        scheme.update({
            "adv_actions": scheme["actions"].copy(),
            "belief": {
                "vshape": (self.args.n_agents,),
                "group": "agents",
                "dtype": th.float32,
            },
            # adversary_id is an episode-level property
            "adversary_id": {
                "vshape": (1,),
                "dtype": th.long,
                "episode_const": True,
            },
        })

        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

        # --- EIR-MAPPO: Initialize adversary attributes that need n_agents ---
        if getattr(self.args, "random_adversary", False):
            self.adapt_adv_probs = np.zeros(self.args.n_agents)

    def _softmax(self, x, axis=None):
        x = x - np.max(x, axis=axis, keepdims=True)
        y = np.exp(x)
        return y / np.sum(y, axis=axis, keepdims=True)

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        # Reset env time variables
        self.t = 0
        self.env_steps_this_run = 0

        self.batch.update({"adversary_id": self.agent_adversary_ids}, ts=0)

        return

    # =========================================================================
    # FGSM 辅助方法
    # =========================================================================
    def _apply_fgsm_perturbation(
        self,
        unterminated_batch,
        envs_not_terminated,
        is_adversary_episodes,
        agent_adversary_ids,
    ):
        """
        对当前时间步 unterminated_batch 中对抗 env 的目标智能体观测施加 FGSM 扰动。

        核心逻辑：
          1. 构建每个未终止 env 的被攻击智能体 ID 列表（非对抗 env 填 -1）。
          2. 调用 mac.compute_fgsm_obs_perturbation() 进行梯度计算，获取扰动后的 obs。
             该方法内部会自动保存/恢复 hidden_states，不污染正式 rollout 的状态。
          3. 将扰动后的 obs 写回 unterminated_batch（当前时间步），使 select_actions
             看到的是扰动观测，并让回放缓冲区存储的也是扰动观测（训练数据一致性）。
          4. 同步回写到 self.batch，保证 episode buffer 里对应位置的数据正确。

        参数：
            unterminated_batch : 当前循环的子批（self.batch[envs_not_terminated]）
            envs_not_terminated : 未终止 env 在全 batch 中的索引列表
            is_adversary_episodes : 全 batch 长度的布尔列表，标记是否为对抗 episode
            agent_adversary_ids  : 全 batch 长度的整数列表，-1 表示无对抗

        返回：
            bool：本时间步是否实际施加了扰动（用于日志统计）
        """
        # 1. 为 unterminated_batch 中每个 env 确定被攻击智能体 ID
        adv_ids_local = [
            agent_adversary_ids[b_idx] if is_adversary_episodes[b_idx] else -1
            for b_idx in envs_not_terminated
        ]

        # 如果当前未终止 batch 里没有任何对抗 env，直接跳过，避免无谓的梯度计算
        if not any(aid >= 0 for aid in adv_ids_local):
            return False

        # 2. 调用 FGSM 计算，需要打开 PyTorch 梯度追踪
        #    compute_fgsm_obs_perturbation 内部会：
        #      - 临时开启 obs 的 requires_grad
        #      - 做一次前向传播（不更新 hidden_states）
        #      - 计算攻击损失并反向传播
        #      - 恢复 hidden_states 和 batch obs
        #      - 返回扰动后的 obs tensor（形状同 batch["obs"][:, t]）
        with th.enable_grad():
            perturbed_obs = self.mac.compute_fgsm_obs_perturbation(
                ep_batch=unterminated_batch,
                t=self.t,
                adv_agent_ids=adv_ids_local,
                envs_not_terminated=envs_not_terminated,
                mac_indices=envs_not_terminated,
                epsilon=getattr(self.args, "fgsm_epsilon", 0.05),
                fgsm_loss_type=getattr(self.args, "fgsm_loss_type", "max_entropy"),
            )

        # perturbed_obs 形状: [len(envs_not_terminated), n_agents, obs_dim]
        # compute_fgsm_obs_perturbation 已在内部恢复了 unterminated_batch["obs"][:, t]
        # 现在将扰动 obs 写入 unterminated_batch 供 select_actions 使用
        unterminated_batch["obs"][:, self.t] = perturbed_obs.detach()

        # 3. 同步回写到 self.batch，保证 episode buffer 中存储的是扰动观测
        #    只更新有对抗智能体的 env 所在的行
        for local_i, b_idx in enumerate(envs_not_terminated):
            if adv_ids_local[local_i] >= 0:
                # self.batch["obs"] 形状: [batch_size, T+1, n_agents, obs_dim]
                self.batch["obs"][b_idx, self.t] = perturbed_obs[local_i].detach()

        return True

    # =========================================================================
    # 主 rollout 函数
    # =========================================================================
    def run(self, test_mode=False):
        agent_adversary_ids = [-1] * self.batch_size
        is_adversary_episodes = [False] * self.batch_size

        train_adversary = getattr(self.args, "random_adversary", False)
        test_adversary = getattr(self.args, "test_adversary", False)

        # -----------------------------------------------------------------
        # 测试期 DLA 参数（已有）
        # test_attack_start_ratio：潜伏期占 episode 的比例，默认 0.4
        # T_switch = episode_limit * test_attack_start_ratio
        # -----------------------------------------------------------------
        test_attack_start_ratio = getattr(self.args, "test_attack_start_ratio", 0.4)
        test_attack_start_t = int(self.episode_limit * test_attack_start_ratio)

        # -----------------------------------------------------------------
        # 训练期 DLA 参数（新增）
        #
        # 配置项（yaml）：
        #   use_dla_training         : bool，是否启用训练期 DLA 潜伏机制，默认 False
        #                              False → 原逻辑（全程对抗，无潜伏）
        #                              True  → DLA 两阶段：前段伪装，后段爆发
        #   train_attack_start_ratio : float，训练期潜伏段占比，默认与测试期一致 (0.4)
        #                              T_switch_train = episode_limit * train_attack_start_ratio
        #
        # 实现的 DLA 定义（对应图片公式 4.13）：
        #   π_adv(a|o, t) = π_coop(a|o),    0 ≤ t < T_switch_train  （伪装，不替换动作）
        #   π_adv(a|o, t) = π*_adv(a|o),    T_switch_train ≤ t ≤ T-1（爆发，替换动作）
        # -----------------------------------------------------------------
        use_dla_training = getattr(self.args, "use_dla_training", False)
        train_attack_start_ratio = getattr(
            self.args, "train_attack_start_ratio", test_attack_start_ratio
        )
        train_attack_start_t = int(self.episode_limit * train_attack_start_ratio)

        victim_interval = getattr(self.args, "victim_interval", 1)
        adv_prob = getattr(self.args, "adv_prob", 0.5)

        for i in range(self.batch_size):
            episode_idx = self.total_episodes_run + i

            if test_mode:
                if test_adversary:
                    is_adversary_episodes[i] = True
                    agent_adversary_ids[i] = int(getattr(self.args, "test_adv_agent_id", 0))
                else:
                    is_adversary_episodes[i] = False
                    agent_adversary_ids[i] = -1

            else:
                if not train_adversary:
                    is_adversary_episodes[i] = False
                    agent_adversary_ids[i] = -1
                    continue

                # 只有在 interval 命中的 episode，才"可能"有攻击
                if (episode_idx % victim_interval) == 0 and (np.random.rand() < adv_prob):
                    is_adversary_episodes[i] = True

                    if getattr(self.args, "adapt_adversary", False):
                        probs = self._softmax(-1 * self.adapt_adv_probs)
                        agent_adversary_ids[i] = int(np.random.choice(self.args.n_agents, p=probs))
                    else:
                        agent_adversary_ids[i] = int(np.random.choice(self.args.n_agents))
                else:
                    is_adversary_episodes[i] = False
                    agent_adversary_ids[i] = -1

        # Store for use in reset()
        self.agent_adversary_ids = [(adv_id,) for adv_id in agent_adversary_ids]

        # Reset envs to run new parallel episodes
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []

        belief_correct_log = []
        adv_episode_count_log = []
        all_rewards = [[] for _ in range(self.batch_size)]

        # =====================================================================
        # FGSM 全局开关：仅在训练期、达到启动时间步后激活
        # 配置项（yaml）：
        #   use_fgsm_training : bool，是否启用训练期 FGSM 扰动，默认 False
        #   fgsm_start_t      : int，从第几个 t_env 步开始施加扰动，默认 0
        #   fgsm_epsilon      : float，扰动强度，默认 0.05
        #   fgsm_loss_type    : str，攻击损失类型，默认 "max_entropy"
        # =====================================================================
        fgsm_enabled = (
            not test_mode
            and getattr(self.args, "use_fgsm_training", False)
            and self.t_env >= getattr(self.args, "fgsm_start_t", 0)
        )
        self._fgsm_applied_steps = 0   # 重置本次 episode 的扰动步数统计

        # =====================================================================
        # 训练期 DLA 统计计数器
        # dla_latent_steps : 本次 run() 中处于潜伏阶段的时间步数（用于日志）
        # dla_burst_steps  : 本次 run() 中处于爆发阶段的时间步数（用于日志）
        # =====================================================================
        _dla_latent_steps = 0
        _dla_burst_steps = 0

        while True:

            # 取出未终止 env 的子批
            unterminated_batch = self.batch[envs_not_terminated]

            is_adv_list = [is_adversary_episodes[b_idx] for b_idx in envs_not_terminated]
            adv_id_list = [agent_adversary_ids[b_idx] for b_idx in envs_not_terminated]

            # =================================================================
            # [FGSM] 在 select_actions 之前对对抗 env 的观测施加扰动
            #
            # 时机选择说明：
            #   - 扰动必须在 select_actions 之前完成，使协作策略看到的是扰动后的观测，
            #     从而让其选出次优动作，模拟真实攻击场景。
            #   - compute_fgsm_obs_perturbation 内部会保存并恢复 hidden_states，
            #     因此不影响后续 select_actions 对 hidden_states 的正常使用。
            #   - perturbed_obs 被写回 unterminated_batch 和 self.batch，
            #     确保经验回放缓冲区存储的是策略实际见到的观测，训练数据一致。
            # =================================================================
            if fgsm_enabled and any(is_adv_list):
                applied = self._apply_fgsm_perturbation(
                    unterminated_batch=unterminated_batch,
                    envs_not_terminated=envs_not_terminated,
                    is_adversary_episodes=is_adversary_episodes,
                    agent_adversary_ids=agent_adversary_ids,
                )
                if applied:
                    self._fgsm_applied_steps += 1

            # =================================================================
            # 正常动作选择（此时 unterminated_batch 已包含扰动观测（如有））
            # =================================================================
            actions, adv_actions, beliefs = self.mac.select_actions(
                unterminated_batch,
                t_ep=self.t,
                t_env=self.t_env,
                test_mode=test_mode,
                is_adversary_episode=is_adv_list,
                agent_adversary_id=adv_id_list,
                mac_indices=envs_not_terminated,
            )

            # 信念准确率统计（仅在 episode 第 0 步、训练期记录）
            if self.t == 0 and not test_mode:
                belief_correct = 0
                adv_episode_count = 0
                for i, b_idx in enumerate(envs_not_terminated):
                    if is_adversary_episodes[b_idx]:
                        adv_episode_count += 1
                        predicted_adv = th.argmax(beliefs[i])
                        if predicted_adv == agent_adversary_ids[b_idx]:
                            belief_correct += 1
                if adv_episode_count > 0:
                    belief_correct_log.append(belief_correct)
                    adv_episode_count_log.append(adv_episode_count)

            # ---  Create input_actions by substituting adversary actions ---
            input_actions = actions.clone()
            for i, b_idx in enumerate(envs_not_terminated):
                if not is_adversary_episodes[b_idx]:
                    continue

                adv_agent_id = int(agent_adversary_ids[b_idx])

                # =============================================================
                # 测试期动作替换逻辑（DLA 两阶段，已有）
                #
                # 阶段一（潜伏期，0 ≤ t < test_attack_start_t）：
                #   pass → input_actions 保持 actions.clone()，即 π_coop
                #   攻击者完美伪装为正常智能体，信念网络无法区分。
                #
                # 阶段二（爆发期，t ≥ test_attack_start_t）：
                #   根据 test_attack_type 替换为对应攻击动作 π*_adv。
                # =============================================================
                if test_mode and getattr(self.args, "test_adversary", False):
                    if self.t >= test_attack_start_t:

                        attack_type = getattr(self.args, "test_attack_type", "random")

                        avail = unterminated_batch["avail_actions"][i, self.t, adv_agent_id]
                        valid = (avail > 0).nonzero(as_tuple=False).squeeze(-1)

                        if valid.numel() == 0:
                            continue

                        game_type = getattr(self.args, "game", "smac")
                        normal_act = int(input_actions[i, adv_agent_id].item())

                        # ------ SMAC ------
                        if game_type == "smac":
                            if attack_type == "random":
                                ridx = th.randint(0, valid.numel(), (1,), device=valid.device).item()
                                input_actions[i, adv_agent_id] = valid[ridx].item()
                            elif attack_type == "policy_adv":
                                input_actions[i, adv_agent_id] = adv_actions[i, adv_agent_id]
                            else:
                                raise ValueError(f"Unknown test_attack_type={attack_type} for SMAC")

                        # ------ MPE ------
                        elif game_type == "mpe":
                            if attack_type == "random":
                                ridx = th.randint(0, valid.numel(), (1,), device=valid.device).item()
                                input_actions[i, adv_agent_id] = valid[ridx].item()
                            elif attack_type == "mirrored":
                                opposite_map = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}
                                if normal_act in opposite_map and opposite_map[normal_act] in valid:
                                    input_actions[i, adv_agent_id] = opposite_map[normal_act]
                                else:
                                    ridx = th.randint(0, valid.numel(), (1,), device=valid.device).item()
                                    input_actions[i, adv_agent_id] = valid[ridx].item()
                            elif attack_type == "malicious_right":
                                input_actions[i, adv_agent_id] = 1
                            else:
                                raise ValueError(f"Unknown test_attack_type={attack_type} for MPE")

                        else:
                            ridx = th.randint(0, valid.numel(), (1,), device=valid.device).item()
                            input_actions[i, adv_agent_id] = valid[ridx].item()
                    else:
                        # 潜伏期：保持 π_coop，不做任何替换
                        pass

                else:
                    # =============================================================
                    # 训练期动作替换逻辑
                    #
                    # 【新增】DLA 两阶段训练支持（use_dla_training=True 时激活）
                    #
                    # 阶段一（潜伏期，0 ≤ t < train_attack_start_t）：
                    #   不替换动作，input_actions 保持 actions.clone() = π_coop。
                    #   攻击者完美伪装，训练数据中包含欺骗性轨迹，
                    #   迫使信念网络和 ABE 模块学习应对信念滞后问题。
                    #
                    # 阶段二（爆发期，t ≥ train_attack_start_t）：
                    #   用对抗策略动作覆盖，= 原有逻辑。
                    #
                    # 【原有】use_dla_training=False：
                    #   全程使用对抗动作，无潜伏期（静态对抗基线行为）。
                    # =============================================================
                    if use_dla_training:
                        if self.t < train_attack_start_t:
                            # DLA 潜伏期：保持 π_coop，不替换
                            # input_actions[i, adv_agent_id] 已是 actions.clone() 的值
                            _dla_latent_steps += 1
                        else:
                            # DLA 爆发期：替换为对抗策略动作 π*_adv
                            input_actions[i, adv_agent_id] = adv_actions[i, adv_agent_id]
                            _dla_burst_steps += 1
                    else:
                        # 原有逻辑：训练期全程使用对抗动作
                        input_actions[i, adv_agent_id] = adv_actions[i, adv_agent_id]

            if getattr(self.args, "debug_adv", False):
                aa = unterminated_batch["avail_actions"][:, self.t]
                for ii in range(input_actions.shape[0]):
                    for a_id in range(input_actions.shape[1]):
                        act = int(input_actions[ii, a_id].item())
                        if aa[ii, a_id, act].item() == 0:
                            raise RuntimeError(
                                f"[INVALID ACTION BEFORE ENV] env_i={ii} agent={a_id} act={act}"
                            )

            cpu_input_actions = input_actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": input_actions.unsqueeze(1),
                "adv_actions": adv_actions.unsqueeze(1),
                "belief": beliefs.unsqueeze(1),
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        parent_conn.send(("step", cpu_input_actions[action_idx]))
                    action_idx += 1

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": [],
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
            }

            # Receive data from each unterminated parallel env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()

                    post_transition_data["reward"].append((data["reward"],))
                    episode_returns[idx] += data["reward"]
                    all_rewards[idx].append(data["reward"])
                    episode_lengths[idx] += 1

                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transition data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data for the next timestep
            # 注意：这里写入的是来自环境的原始真实观测，而非扰动观测。
            # 下一时间步循环开始时，若 FGSM 仍激活，会再次对该时间步施加扰动。
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        # =====================================================================
        # FGSM 统计日志（训练期）
        # =====================================================================
        if not test_mode and fgsm_enabled and self._fgsm_applied_steps > 0:
            self.logger.log_stat("fgsm_applied_steps", self._fgsm_applied_steps, self.t_env)
            if self.json_logger is not None:
                self.json_logger.log_stat("fgsm_applied_steps", self._fgsm_applied_steps, self.t_env)

        # =====================================================================
        # 训练期 DLA 统计日志（新增）
        # 记录本次 run() 中潜伏/爆发步数，便于验证 DLA 机制是否正确激活
        # =====================================================================
        if not test_mode and use_dla_training:
            if _dla_latent_steps > 0 or _dla_burst_steps > 0:
                self.logger.log_stat("dla_latent_steps", _dla_latent_steps, self.t_env)
                self.logger.log_stat("dla_burst_steps", _dla_burst_steps, self.t_env)
                if self.json_logger is not None:
                    self.json_logger.log_stat("dla_latent_steps", _dla_latent_steps, self.t_env)
                    self.json_logger.log_stat("dla_burst_steps", _dla_burst_steps, self.t_env)

        if test_mode:
            self.test_steps_total += int(sum(episode_lengths))
            self.test_episode_idx += int(self.batch_size)

        if not test_mode:
            self.t_env += self.env_steps_this_run
            self.total_episodes_run += self.batch_size

        # Get stats back for each parallel env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))
        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = final_env_infos
        if len(infos) > 0:
            cur_stats.update({k: sum(d.get(k, 0) for d in infos)
                              for k in set.union(*[set(d) for d in infos])})

        cur_stats["n_episodes"] = cur_stats.get("n_episodes", 0) + self.batch_size
        cur_stats["ep_length"] = cur_stats.get("ep_length", 0) + sum(episode_lengths)

        if not test_mode:
            if sum(adv_episode_count_log) > 0:
                cur_stats["belief_accuracy"] = sum(belief_correct_log) / sum(adv_episode_count_log)
            else:
                cur_stats["belief_accuracy"] = 0
            cur_stats["n_adv_episodes"] = sum(adv_episode_count_log)

        if test_mode:
            cur_returns.extend(episode_returns)
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

        if self.txt_logger is not None and prefix == "test_":
            self.txt_logger.log(self.t_env, np.mean(returns))

        if self.json_logger is not None:
            self.json_logger.log_stat(prefix + "return_target_mean", np.mean(returns), self.t_env)
            self.json_logger.log_stat(prefix + "return_target_std", np.std(returns), self.t_env)

        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, terminated, env_info = env.step(actions)
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "info": env_info,
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs(),
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
