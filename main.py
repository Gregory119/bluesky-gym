"""
This file is an example train and test loop for the different environments.
Selecting different environments is done through setting the 'env_name' variable.

TODO:
* add rgb_array rendering for the different environments to allow saving videos
"""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG

import numpy as np

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

from pathlib import Path
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise


bluesky_gym.register_envs()


class EpsilonDecayScheduler():
    # linearly decays the epsilon from start to finish over the aneal time.
    def __init__(self, eps_start, eps_finish, eps_aneal_time):
        self.eps_start = eps_start
        self.eps_finish = eps_finish
        self.eps_aneal_time = eps_aneal_time
        self.dd = (self.eps_finish - self.eps_start) / self.eps_aneal_time

    def get_epsilon(self, current_time_step):
        if self.dd < 0:
            return max(self.eps_finish, self.eps_start + self.dd * current_time_step)
        else:
            return min(self.eps_finish, self.eps_start + self.dd * current_time_step)


class UniformEGreedyNoise(ActionNoise):
    # create action noise with a uniform distribution of the airspace, and
    # sample this distribution with a linearly decaying epsilon greedy policy
    
    def __init__(self) -> None:
        self.decay_scheduler = EpsilonDecayScheduler(eps_start=0.5, eps_finish=0., eps_aneal_time=250e3)
        self.timestep = 0
        super().__init__()

    def __call__(self) -> np.ndarray:
        self.timestep += 1
        epsilon = self.decay_scheduler.get_epsilon(self.timestep)
        if np.random.rand() < epsilon:
            # there are 3 action elements so this needs have the same no. of elements
            return np.random.uniform(low=-1.0, high=1.0, size=3)
        return np.array([0.0, 0.0, 0.0])


class LoggerCallback(BaseCallback):
    # used to log episode rewards to tensorboard
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.dones_size = 0

    def _on_step(self) -> bool:
        # log each episode reward
        if self.locals['dones'][0]:
            # episode termination detected
            self.logger.record("episode_reward", self.locals['infos'][0]['total_reward'])
            self.logger.dump(self.num_timesteps)
        return True


#env_name = 'SectorCREnv-v0'
env_name = 'SectorCREnv-v1'
#env_name = 'StaticObstacleEnv-v0'
#env_name = 'StaticObstacleEnv-v1'
#env_name = 'VerticalCREnv-v0'
algorithm = PPO
alg_name = str(algorithm.__name__)

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{alg_name}.csv'

TRAIN = False
EVAL_EPISODES = 100
VERBOSE = False


if __name__ == "__main__":
    #env = gym.make(env_name, render_mode='human')
    env = gym.make(env_name, render_mode=None)
    obs, info = env.reset()
    
    if env_name.startswith('SectorCREnv') and alg_name == "PPO":
        model = algorithm("MultiInputPolicy",
                          env,
                          verbose=VERBOSE,
                          learning_rate=3e-4,
                          gae_lambda=0.05,
                          ent_coef=0.01,
                          tensorboard_log=f"./logs/{env_name}_{alg_name}")
    elif env_name.startswith('StaticObstacleEnv') and alg_name == "PPO":
        model = algorithm("MultiInputPolicy",
                          env,
                          verbose=VERBOSE,
                          learning_rate=3e-4,
                          clip_range=0.2,
                          ent_coef=0.02,
                          tensorboard_log=f"./logs/{env_name}_{alg_name}")
    elif env_name.startswith('SectorCREnv') and alg_name == "DDPG":
        model = algorithm("MultiInputPolicy",
                          env,
                          verbose=VERBOSE,
                          learning_rate=1e-4,
                          tau=0.001,
                          action_noise=UniformEGreedyNoise(),
                          tensorboard_log=f"./logs/{env_name}_{alg_name}")
    elif env_name.startswith('StaticObstacleEnv') and alg_name == "DDPG":
        model = algorithm("MultiInputPolicy",
                          env,
                          verbose=VERBOSE,
                          learning_rate=1e-5,
                          tau=1e-3,
                          action_noise=UniformEGreedyNoise(),
                          tensorboard_log=f"./logs/{env_name}_{alg_name}")

    if TRAIN:
        model.learn(total_timesteps=500e3, callback=LoggerCallback(), progress_bar=True)
        model.save(f"models/{env_name}/{env_name}_{alg_name}/model")
        del model
    env.close()
    print("training complete")
    
    # Test the trained model
    model = algorithm.load(f"models/{env_name}/{env_name}_{alg_name}/model", env=env)
    env = gym.make(env_name, render_mode="human")
    #env = gym.make(env_name, render_mode=None)
    for i in range(EVAL_EPISODES):

        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            tot_rew += reward
        print(tot_rew)
    env.close()
