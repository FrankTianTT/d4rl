import numpy as np
import random
from gym import Env
from gym.spaces import Box
import math
from d4rl import offline_env
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


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


def line(x, x1, x2, y1, y2):
    return y2 - (x2 - x) * (y2 - y1) / (x2 - x1)


class ParticleEnv(Env):
    """
    Toy environment for causal discovery and offline reinforcement learning.
    """

    def __init__(self,
                 init=(0, 0),
                 goal=(100, 100),
                 region=1e3,
                 v_dist_boundary=35,
                 v_max=4,
                 return_direction=True,
                 return_velocity=True):
        self.goal = goal
        self.init = init
        self.region = region
        self.v_dist_boundary = v_dist_boundary
        self.v_max = v_max
        self.position = list(self.init)
        self.direction = 0

        self.v = 0.5
        self.vx = 0
        self.vy = 0

        self.return_direction = return_direction
        self.return_velocity = return_velocity

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
        self.v = self.get_v(calculate_distance(self.position, self.init))

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


def find_v_dist_boundary(num=int(2e5), max_episode_steps=300):
    v_dist_boundary = 40
    env = TimeLimit(ParticleEnv(v_dist_boundary=v_dist_boundary), max_episode_steps)
    episode_rewards = []

    l = []

    t = 0
    episode_reward = 0
    obs = env.reset()
    while t < num:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        l.append(calculate_distance(env.position, env.init) < v_dist_boundary)

        if done:
            obs = env.reset()

        t += 1

    print(Counter(l)[False] / num)


def get_particle_env(**kwargs):
    return OfflineParticleEnv(**kwargs)


if __name__ == "__main__":
    # env = get_particle_env()
    # env.draw_v_curve()

    find_v_dist_boundary()