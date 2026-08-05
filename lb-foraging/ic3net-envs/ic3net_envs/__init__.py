from gym.envs.registration import register

register(
    id='PredatorPrey-v0',
    entry_point='ic3net_envs.predator_prey_env:PredatorPreyEnv',
)

register(
    id="TrafficJunction-5p-1v-easy-v0",
    entry_point="ic3net_envs.traffic_junction_env:TrafficJunctionEnv",
    kwargs={
        "nagents": 5,
        "difficulty": "easy",
        "vision": 1
    },
)
