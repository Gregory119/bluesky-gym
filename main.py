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


bluesky_gym.register_envs()

#env_name = 'SectorCREnv-v0'
env_name = 'SectorCREnv-v1'
#env_name = 'StaticObstacleEnv-v1'
#env_name = 'VerticalCREnv-v0'
algorithm = PPO
alg_name = str(algorithm.__name__)

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{alg_name}.csv'

TRAIN = True
EVAL_EPISODES = 10
VERBOSE = False


class LoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.dones_size = 0

    def _on_step(self) -> bool:
        # log each episode reward
        if self.locals['dones'][0]:
            # episode termination detected
            #print("total reward: {}".format(self.locals['infos'][0]['total_reward']))
            self.logger.record("episode_reward", self.locals['infos'][0]['total_reward'])
            self.logger.dump(self.num_timesteps)
        return True


if __name__ == "__main__":
    # env = gym.make(env_name, render_mode='human')
    env = gym.make(env_name, render_mode=None)
    obs, info = env.reset()
    model = algorithm("MultiInputPolicy", env, verbose=VERBOSE, learning_rate=3e-4, tensorboard_log=f"./logs/{env_name}_{alg_name}")
    if TRAIN:
        model.learn(total_timesteps=2e6, callback=LoggerCallback(), progress_bar=True)
        model.save(f"models/{env_name}/{env_name}_{alg_name}/model")
        del model
    env.close()
    print("training complete")
    
    # Test the trained model
    model = algorithm.load(f"models/{env_name}/{env_name}_{alg_name}/model", env=env)
    #env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name, render_mode=None)
    for i in range(EVAL_EPISODES):

        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            # action = np.array(np.random.randint(-100,100,size=(2))/100)
            # action = np.array([0,-1])
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward
        print(tot_rew)
    env.close()
