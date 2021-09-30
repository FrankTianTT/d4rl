import numpy as np
import random
from gym import Env
from gym.spaces import Box
import math
from d4rl import offline_env
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from d4rl.gym_particle.util import *


class ParticleEnv(Env):
    """
    Toy environment for causal discovery and offline reinforcement learning.
    """

    def __init__(self,
                 init=(0, 0),
                 goal=(100, 100),
                 region=1e3,
                 v_dist_boundary=30,
                 v_max=4):
        self.init = init
        self.goal = goal
        self.region = region
        self.v_dist_boundary = v_dist_boundary
        self.v_max = v_max

        self.last_state = []
        self.state = []

        self.action_space = Box(np.array([-100]), np.array([100]))
        state_low = [0, -0.1, -0.1, -0.1, -self.region, -self.region]
        state_high = [math.pi * 2, 0.1, 0.1, 0.1, self.region, self.region]
        self.observation_space = Box(np.array(state_low), np.array(state_high))

    def step(self, action):
        action = action[0]
        self.update_state(action)

        reward = self.get_reward()
        done = calculate_distance(self.state[-2:], self.init) > self.region
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.reset_state()
        return np.array(self.state)

    def render(self, mode='human'):

        print("x:{:.2f}\ty:{:.2f}".format(*self.state[-2:]))

    def reset_state(self):
        self.state = []
        px, py = list(np.array(self.init) + (np.random.rand(2) - 0.5) * 5)
        direction = (random.random() * (2 * math.pi)) % (2 * math.pi)

        v = self.get_v(calculate_distance([px, py], self.init))
        vx, vy = v * math.cos(direction), v * math.sin(direction)
        self.state = [direction, v, vx, vy, px, py]

    def update_state(self, action):
        self.last_state = self.state

        last_direction, last_v, last_vx, last_vy, last_px, last_py = self.last_state
        angle = tanh(action, t=100, alpha=math.pi)
        direction = (last_direction + angle) % (2 * math.pi)
        v = self.get_v(calculate_distance([last_px, last_py], self.init))
        vx, vy = last_v * math.cos(direction), last_v * math.sin(direction)
        px = last_px + last_vx
        py = last_py + last_vy

        self.state = [direction, v, vx, vy, px, py]

    def get_reward(self):
        last_distance = calculate_distance(self.last_state[-2:], self.goal)
        distance = calculate_distance(self.state[-2:], self.goal)
        distance_diff = distance - last_distance
        return - distance_diff

    def get_v(self, dis):
        if dis < self.v_dist_boundary:
            v = line(dis, 0, self.v_dist_boundary, 0.5, self.v_max)
        else:
            v = (1.5 - sigmoid(dis - self.v_dist_boundary, t=5)) * self.v_max

        return v

    def draw_v_curve(self):
        x = np.arange(100)
        y = [self.get_v(d) for d in x]
        plt.plot(x, y)
        plt.show()


class OfflineParticleEnv(ParticleEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        ParticleEnv.__init__(self)
        offline_env.OfflineEnv.__init__(self, **kwargs)


def find_v_dist_boundary(num=int(2e5), max_episode_steps=100):
    v_dist_boundary = 30
    env = TimeLimit(ParticleEnv(v_dist_boundary=v_dist_boundary), max_episode_steps)
    episode_rewards = []

    l = []

    t = 0
    episode_reward = 0
    obs = env.reset()
    while t < num:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        l.append(calculate_distance(env.state[-2:], env.init) < v_dist_boundary)

        if done:
            obs = env.reset()

        t += 1

    print(Counter(l)[False] / num)


def get_particle_env(**kwargs):
    return OfflineParticleEnv(**kwargs)


if __name__ == "__main__":
    find_v_dist_boundary()
