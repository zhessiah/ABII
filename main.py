import numpy as np
import os
import random
from collections.abc import Mapping
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.coustom_logging import get_logger
import yaml
import pickle

from run import run

os.environ["SC2PATH"] ="/home/waq/SC2/StarCraftII/"

# 启用 PyTorch 的异常侦测器，用于捕获 NaN 值的来源
th.autograd.set_detect_anomaly(True)

from gym.envs.registration import register as gym_register, registry as gym_registry

def register(id, entry_point, kwargs):
    
    # Avoid overriding if the environment id is already registered
    try:
        # Gym <=0.21
        env_specs = getattr(gym_registry, "env_specs", None)
        if env_specs is not None and id in env_specs:
            return
        # Fallback for other registry shapes
        if env_specs is None:
            try:
                # Some versions expose a mapping interface
                if id in gym_registry.keys():
                    return
            except Exception:
                pass
    except Exception:
        pass
    return gym_register(id=id, entry_point=entry_point, kwargs=kwargs)

register(
    id="Foraging-8x8-5p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-12x12-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 3,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 11,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-4f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-15x15-3p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8s-25x25-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (25, 25),
        "max_food": 5,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5s-25x25-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (25, 25),
        "max_food": 5,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-50x50-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (50, 50),
        "max_food": 5,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-30x30-7p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 4,
        "sight": 30,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-30x30-7p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 5,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-30x30-7p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 4,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-30x30-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 5,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-15x15-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-9x9-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-8x8-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-6x6-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (6, 6),
        "max_food": 2,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-6x6-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (6, 6),
        "max_food": 1,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 1,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 2,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-7x7-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-4p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-4p-2f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-8x8-4p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 2,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 1,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-6p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 6,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-4f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)


register(
    id="Foraging-10x10-4p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 1,
        "sight": 10,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)


register(
    id="Foraging-9x9-4p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-20x20-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5s-20x20-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 3,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-13x13-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 13,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 11,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 3,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 3,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-10x10-3p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 3,
        "sight": 10,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-6s-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 12,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)


register(
    id="Foraging-8x8-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-5x5-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 1,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

# 设置Sacred实验框架的捕获模式为系统模式，避免文件描述符超时
SETTINGS['CAPTURE_MODE'] = "sys"
logger = get_logger()

# 创建Sacred实验实例
ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

# 设置结果保存路径
results_path = os.path.join(dirname(abspath(__file__)), "a2r_results")

@ex.main
def my_main(_run, _config, _log):
    """主实验函数
    Args:
        _run: Sacred实验运行实例
        _config: 配置参数
        _log: 日日志记录器
    """
    # 打印CUDA和GPU信息
    _log.info(f"PyTorch CUDA available: {th.cuda.is_available()}")
    if th.cuda.is_available():
        _log.info(f"Current CUDA device: {th.cuda.current_device()}")
        _log.info(f"Device name: {th.cuda.get_device_name(th.cuda.current_device())}")

    # 设置随机种子，确保实验可重现性
    config = config_copy(_config)
    np.random.seed(config["seed"])  # numpy随机种子
    th.manual_seed(config["seed"])  # PyTorch随机种子
    config['env_args']['seed'] = config["seed"]  # 环境随机种子
    config["sacred_path"] = _run.observers[0].dir
    # 运行训练框架
    run(_run, config, _log)

    # 在实验成功结束后打印提示信息
    _log.info("Experiment finished successfully.")
    print("\n" + "="*50)
    print("      Experiment finished successfully!      ")
    print("="*50 + "\n")


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    params = deepcopy(sys.argv)
    
    gpu_ids = None
    for _i, _v in enumerate(params):
        if _v.startswith("--gpu="):
            # 支持传入多个GPU，例如 --gpu=0,1,2
            gpu_ids = _v.split("=")[1]
            del params[_i]
            break
    
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"Running on GPUs: {gpu_ids}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print("No GPU specified, defaulting to GPU 1.")
    
    th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]    
    
    # ========== 新增自定义层级参数解析：--exp-layer=自定义名称 ==========
    exp_layer = None
    for _i, _v in enumerate(params):
        if _v.startswith("--exp-layer="):
            exp_layer = _v.split("=")[1]
            del params[_i]
            break
    
    # now add all the config to sacred
    ex.add_config(config_dict)
    ex.add_config({"exp_layer": exp_layer})
    
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    # 构建结果路径：支持新增自定义层级
    file_obs_path = os.path.join(results_path, f"sacred/{config_dict['game']}")
    # 如果传入了自定义层级，添加该层级
    if exp_layer is not None:
        file_obs_path = os.path.join(file_obs_path, exp_layer)
    # 添加map_name层级
    file_obs_path = os.path.join(file_obs_path, map_name)
    
    # 针对mpe游戏，补充num_agents层级（保留原逻辑）
    if config_dict['game'] == 'mpe':
        num_agents =  env_config['env_args']['num_agents']
        file_obs_path = os.path.join(file_obs_path, f"Add/num_agent_{num_agents}" )

    ex.observers.append(FileStorageObserver.create(file_obs_path))
    # ex.observers.append(MongoObserver())

    ex.run_commandline(params)