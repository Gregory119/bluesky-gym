# RBE-595 Reinforcement Learning Project

## Dependencies
This is a fork of the bluesky-gym code with my added changes. It contains a file
called `pyproject.toml` that has a list of the require dependencies under the
`[tool.poetry.dependencies]` key. If you have poetry installed then you can use
this file to install the dependencies easily.

## New/Modified Files
### Environments
New files were created for the environments under the sub-directory
`bluesky_gym/envs`. These include:
- `sector_cr_env_v1.py` for the sector environment
- `static_obstacle_env_v1.py` for the static obstacle environment

The original environment source code was copied and then modified.

### Main Script
The top-level `main.py` script was update to train and test the new environments
and includes some additional code for logging and adding noise to DDPG
actions. The `env_name` variable can be set to `SectorCREnv-v1` or
`StaticObstacleEnv-v1` to select either the new sector or static obstacle
environment, respectively. The algorithm can then be selected by setting the
`algorithm` variable to either `PPO` or `DDPG`.
