import gym
from gym import spaces
from gym.spaces import Discrete

from gym.utils import seeding

import numpy as np
from os import path
import math

from typing import Tuple


class InValidActionException (Exception):
    pass


class RandomMaze(gym.Env):
    '''Constructor. Takes number of states as a parameter.

        States are: 

        0        1        2         3(Goal)
        4        5(Wall)  6         7(Hole)
        8(Start) 9        10        11

        Action space :
        0 -> LEFT
        1 -> UP
        2 -> RIGHT
        3 -> DOWN

        Allocate the Reward:
        If Goal state: +1
        If Hole state: -1
        Otherwise: -0.04
    '''

    def __init__(self):
        self.start_state = int(8)
        self.goal = int(3)
        self.hole = int(7)
        self.state = int(8)
        self.transition_matrix = {
            0: {
                0: [(0.8, 0, -0.04, False), (0.1, 0, -0.04, False), (0.1, 4, -0.04, False)],
                1: [(0.8, 0, -0.04, False), (0.1, 0, -0.04, False), (0.1, 1, -0.04, False)],
                2: [(0.8, 1, -0.04, False), (0.1, 0, -0.04, False), (0.1, 4, -0.04, False)],
                3: [(0.8, 4, -0.04, False), (0.1, 0, -0.04, False), (0.1, 1, -0.04, False)],
            },
            1: {
                0: [(0.8, 0, -0.04, False), (0.1, 1, -0.04, False), (0.1, 1, -0.04, False)],
                1: [(0.8, 1, -0.04, False), (0.1, 0, -0.04, False), (0.1, 2, -0.04, False)],
                2: [(0.8, 2, -0.04, False), (0.1, 1, -0.04, False), (0.1, 1, -0.04, False)],
                3: [(0.8, 1, -0.04, False), (0.1, 0, -0.04, False), (0.1, 2, -0.04, False)],
            },
            2: {
                0: [(0.8, 1, -0.04, False), (0.1, 2, -0.04, False), (0.1, 6, -0.04, False)],
                1: [(0.8, 2, -0.04, False), (0.1, 1, -0.04, False), (0.1, 3, +1, True)],
                2: [(0.8, 3, +1, True), (0.1, 2, -0.04, False), (0.1, 6, -0.04, False)],
                3: [(0.8, 6, -0.04, False), (0.1, 1, -0.04, False), (0.1, 3, +1, True)],
            },
            3: {
                0: [(0.8, 3, 0, True), (0.1, 3, 0, True), (0.1, 3, 0, True)],
                1: [(0.8, 3, 0, True), (0.1, 3, 0, True), (0.1, 3, 0, True)],
                2: [(0.8, 3, 0, True), (0.1, 3, 0, True), (0.1, 3, 0, True)],
                3: [(0.8, 3, 0, True), (0.1, 3, 0, True), (0.1, 3, 0, True)],
            },
            4: {
                0: [(0.8, 4, -0.04, False), (0.1, 0, -0.04, False), (0.1, 8, -0.04, False)],
                1: [(0.8, 0, -0.04, False), (0.1, 4, -0.04, False), (0.1, 4, -0.04, False)],
                2: [(0.8, 4, -0.04, False), (0.1, 0, -0.04, False), (0.1, 8, -0.04, False)],
                3: [(0.8, 8, -0.04, False), (0.1, 4, -0.04, False), (0.1, 4, -0.04, False)],
            },
            6: {
                0: [(0.8, 6, -0.04, False), (0.1, 2, -0.04, False), (0.1, 10, -0.04, False)],
                1: [(0.8, 2, -0.04, False), (0.1, 6, -0.04, False), (0.1, 7, -1, True)],
                2: [(0.8, 7, -1, True), (0.1, 2, -0.04, False), (0.1, 10, -0.04, False)],
                3: [(0.8, 10, -0.04, False), (0.1, 6, -0.04, False), (0.1, 7, -1, True)],
            },
            7: {
                0: [(0.8, 7, 0, True), (0.1, 7, 0, True), (0.1, 7, 0, True)],
                1: [(0.8, 7, 0, True), (0.1, 7, 0, True), (0.1, 7, 0, True)],
                2: [(0.8, 7, 0, True), (0.1, 7, 0, True), (0.1, 7, 0, True)],
                3: [(0.8, 7, 0, True), (0.1, 7, 0, True), (0.1, 7, 0, True)],
            },
            8: {
                0: [(0.8, 8, -0.04, False), (0.1, 8, -0.04, False), (0.1, 4, -0.04, False)],
                1: [(0.8, 4, -0.04, False), (0.1, 8, -0.04, False), (0.1, 9, -0.04, False)],
                2: [(0.8, 9, -0.04, False), (0.1, 8, -0.04, False), (0.1, 4, -0.04, False)],
                3: [(0.8, 8, -0.04, False), (0.1, 8, -0.04, False), (0.1, 9, -0.04, False)],
            },
            9: {
                0: [(0.8, 8, -0.04, False), (0.1, 9, -0.04, False), (0.1, 9, -0.04, False)],
                1: [(0.8, 9, -0.04, False), (0.1, 8, -0.04, False), (0.1, 10, -0.04, False)],
                2: [(0.8, 10, -0.04, False), (0.1, 9, -0.04, False), (0.1, 9, -0.04, False)],
                3: [(0.8, 9, -0.04, False), (0.1, 8, -0.04, False), (0.1, 10, -0.04, False)]
            },
            10: {
                0: [(0.8, 9, -0.04, False), (0.1, 6, -0.04, False), (0.1, 10, -0.04, False)],
                1: [(0.8, 6, -0.04, False), (0.1, 9, -0.04, False), (0.1, 11, -0.04, False)],
                2: [(0.8, 11, -0.04, False), (0.1, 6, -0.04, False), (0.1, 10, -0.04, False)],
                3: [(0.8, 10, -0.04, False), (0.1, 9, -0.04, False), (0.1, 11, -0.04, False)]
            },
            11: {
                0: [(0.8, 10, -0.04, False), (0.1, 7, -1, True), (0.1, 11, -0.04, False)],
                1: [(0.8, 7, -1, True), (0.1, 10, -0.04, False), (0.1, 11, -0.04, False)],
                2: [(0.8, 11, -0.04, False), (0.1, 7, -1, True), (0.1, 11, -0.04, False)],
                3: [(0.8, 11, -0.04, False), (0.1, 11, -0.04, False), (0.1, 10, -0.04, False)]
            }
        }

    def seed(self, seed):
        np.random.seed(seed)
        return seed

    def step(self, action):
        d = np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
        tr = self.transition_matrix[int(self.state)][int(action)][int(d)]
        prob = tr[0]
        self.state = int(tr[1])
        reward = tr[2]
        is_Terminal = tr[3]
        return self.state, reward, is_Terminal, {}

    def reset(self):
        self.state = int(8)
        return self.state, False
