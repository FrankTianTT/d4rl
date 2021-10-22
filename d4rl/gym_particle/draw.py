from stable_baselines3 import SAC
from particle import ParticleEnv
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from d4rl.gym_particle.util import *

max_episode_steps = 100
model = SAC.load("logs/rl_model_100000_steps.zip")


def draw(policy, num):
    env = TimeLimit(ParticleEnv(), max_episode_steps)
    rewards = []

    for i in tqdm(range(num)):
        x = []
        y = []
        rewards.append(0)
        obs = env.reset()
        while True:
            x.append(obs[-2])
            y.append(obs[-1])

            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            rewards[-1] += reward

            if done:
                break

        plt.plot(x, y)
    plt.show()

    print(sum(rewards) / num)


def sac_policy(obs):
    action, state = model.predict(obs, deterministic=True)
    return action


def random_policy(obs):
    return (np.random.random(1) - 0.5) * 200


def expert_policy(obs):
    direction, v, vx, vy, px, py = obs
    goal_direction = math.atan((100 - py) / (100 - px))
    if px > 100 and py > 100:
        goal_direction += math.pi
    angle = goal_direction - direction
    action = artanh(angle, t=100, alpha=math.pi)
    return [action]


def try_policy(obs):
    return [0]


if __name__ == '__main__':
    # draw(expert_policy)
    # draw(random_policy)
    draw(sac_policy, 5)

