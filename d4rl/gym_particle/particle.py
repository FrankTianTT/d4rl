import numpy as np
import random
from gym import Env
from gym.spaces import Box
import math
from d4rl import offline_env
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
from collections import defaultdict


def sigmoid(x, t=1e3, alpha=1.):
    x = x / t
    y = 1 / (1 + math.exp(-x / t))
    return alpha * y


def tanh(x, t=1., alpha=1.):
    x = x / t
    y = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    return alpha * y


def calculate_distance(pos, goal):
    return ((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2) ** 0.5


class ParticleEnv(Env):
    """
    Toy environment for causal discovery and offline reinforcement learning.
    """

    def __init__(self, init=(0, 0), goal=(100, 100), return_direction=True, return_velocity=True, region=1e3):
        self.goal = goal
        self.init = init
        self.position = list(self.init)
        self.direction = 0
        self.v = 0.5
        self.vx = 0
        self.vy = 0

        self.return_direction = return_direction
        self.return_velocity = return_velocity

        self.region = region

        self._build_space()

    def _build_space(self):
        self.action_space = Box(np.array([-100]), np.array([100]))
        state_low = []
        state_high = []
        if self.return_direction:
            state_low.append(0)
            state_high.append(math.pi * 2)
        if self.return_velocity:
            state_low.append(-0.1)
            state_high.append(0.1)
        state_low.extend([-0.1, -0.1, -self.region, -self.region])
        state_high.extend([0.1, 0.1, self.region, self.region])
        self.observation_space = Box(np.array(state_low), np.array(state_high))

    def step(self, action):
        action = tanh(action[0], t=100, alpha=math.pi)
        # action = action[0]
        self.direction = (self.direction + action) % (2 * math.pi)

        state = self._get_state()
        reward = 1 - calculate_distance(self.position, self.goal) / calculate_distance(self.init, self.goal)
        done = calculate_distance(self.position, self.init) > self.region
        return state, reward, done, {}

    def set_state(self, state):
        assert len(state) == 6, "must 6 dim"
        self.direction, self.v, self.vx, self.vy, self.position[0], self.position[1] = list(state)

    def reset(self):
        self.position = list(np.array(self.init) + (np.random.rand(2) - 0.5) * 5)
        self.direction = (random.random() * (2 * math.pi)) % (2 * math.pi)
        return self._get_state()

    def render(self, mode='human'):
        print("x:{:.2f}\ty:{:.2f}".format(*self.position))

    def _get_state(self):
        self.v = 1 if calculate_distance(self.position, self.init) < 35 else 2

        self.position[0] += self.vx
        self.position[1] += self.vy

        self.vx, self.vy = self.v * math.cos(self.direction), self.v * math.sin(self.direction)

        state = []
        if self.return_direction:
            state.append(self.direction)
        if self.return_velocity:
            state.append(self.v)
        state.extend([self.vx, self.vy])
        state.extend(self.position)
        state = np.array(state)
        return state


class OfflineParticleEnv(ParticleEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        ParticleEnv.__init__(self)
        offline_env.OfflineEnv.__init__(self, **kwargs)


def get_particle_env(**kwargs):
    return OfflineParticleEnv(**kwargs)


if __name__ == "__main__":
    import d4rl
    env = get_particle_env(dataset_url='http://114.212.21.162:6868/offline_dataset/particle-medium-replay-v0.hdf5')
    d = d4rl.qlearning_dataset(env)
    idx = 200
    s, next_s, a, r = d['observations'][idx], d['next_observations'][idx], d['actions'][idx], d['rewards'][idx]
    print(s, a)

    print()

    print(next_s, r)

    env.set_state(s)
    new_obs, reward, done, _ = env.step(a)
    print(new_obs, [reward])

    # [  5.049894     1.           0.33113363  -0.94358385   7.8868127
    #  -12.708135  ] -0.029270684
    # [  4.57361413   1.          -0.13832985  -0.99038621   7.88681269
    #  -12.70813465] [-0.029270685814085873]

    # what's wrong with u?
