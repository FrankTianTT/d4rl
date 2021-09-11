from stable_baselines3 import SAC
from particle import ParticleEnv
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
import numpy as np


max_episode_steps = 300
model = SAC.load("save/best_model.zip")


def draw(policy):
    env = TimeLimit(ParticleEnv(), max_episode_steps)
    rewards = 0

    x = []
    y = []
    obs = env.reset()
    while True:
        x.append(obs[-2])
        y.append(obs[-1])
        print(env.v)

        action = policy(obs)
        obs, reward, done, _ = env.step(action)
        rewards += reward

        if done:
            break

    plt.plot(x, y)
    plt.show()

    print(rewards)


def sac_policy(obs):
    action, state = model.predict(obs, deterministic=True)
    return action


def random_policy(obs):
    return (np.random.random(1) - 0.5) * 200


def expert_policy(obs):
    return


if __name__ == '__main__':
    draw(sac_policy)
