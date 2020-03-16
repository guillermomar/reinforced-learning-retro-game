import retro
import numpy as np

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR
from stable_baselines import PPO2
from stable_baselines import A2C

def make_env(env_id, state, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = retro.make(env_id,state)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    env_id = "StreetFighterIISpecialChampionEdition-Genesis"
    state = "Champion.Level1.RyuVsGuile"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, state, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    # model = A2C(CnnPolicy, env, verbose=1)
    # model.learn(total_timesteps=1000000)
    # model.save("sf2-a2c")

    model = A2C.load("sf2-a2c")
    

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        # print(action)
        obs, rewards, dones, info = env.step(action)
        # print(rewards)
        env.render()