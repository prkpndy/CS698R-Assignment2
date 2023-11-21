'''Register all environments with gym.

Needs to be updated whenever a new environment is added.
'''
from gym.envs.registration import register

register(
    id='random_maze-v0',
    entry_point='environments.envs:RandomMaze',
)
