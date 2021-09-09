import gym
import os
import torch
import stable_baselines3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3 import SAC
from particle import ParticleEnv
from gym.wrappers import TimeLimit
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from collections import defaultdict
import h5py
from tqdm import tqdm

max_episode_steps = 300


def train(total_timesteps=int(2e5)):
    eval_env = TimeLimit(ParticleEnv(), max_episode_steps)
    env = TimeLimit(ParticleEnv(), max_episode_steps)

    eval_callback = EvalCallback(eval_env, best_model_save_path="save",
                                 log_path="save", eval_freq=int(5e3),
                                 deterministic=True, render=True)

    model = SAC('MlpPolicy', env, tensorboard_log="./log",
                batch_size=256)

    model.learn(total_timesteps, callback=eval_callback)
    return collect_offline_data_from_model(model)


def collect_offline_data_from_model(model):
    replay_buffer = model.replay_buffer
    pos = replay_buffer.pos
    samples = {'observations': replay_buffer.observations[:pos].reshape(pos, -1),
               'actions': replay_buffer.actions[:pos].reshape(pos, -1),
               'rewards': replay_buffer.rewards[:pos].reshape(pos),
               'terminals': replay_buffer.dones[:pos].reshape(pos),
               'timeouts': replay_buffer.timeouts[:pos].reshape(pos)}

    return samples


def draw():
    env = TimeLimit(ParticleEnv(), max_episode_steps)
    model = SAC.load("save/best_model.zip")

    rewards = 0

    x = []
    y = []
    obs = env.reset()
    while True:
        x.append(obs[-2])
        y.append(obs[-1])

        action, state = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        rewards += reward

        if done:
            break
    plt.plot(x, y)
    plt.show()

    print(rewards)


def collect_offline_data(num=int(2e5), policy="random"):
    env = TimeLimit(ParticleEnv(), max_episode_steps)
    episode_rewards = []

    t = 0
    episode_reward = 0
    obs = env.reset()
    samples = defaultdict(list)
    model = SAC.load("save/best_model.zip")
    while t < num:
        if policy == "random":
            action = env.action_space.sample()
        else:
            action, state = model.predict(obs, deterministic=True)

        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward

        samples['observations'].append(obs)
        samples['actions'].append(action)
        samples['rewards'].append(reward)
        samples['terminals'].append(float(done))
        samples['timeouts'].append(float(0))

        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
        else:
            obs = next_obs

        t += 1

    np_samples = {}
    for key in samples.keys():
        np_samples[key] = np.array(samples[key])

    return np_samples, min(episode_rewards), max(episode_rewards)


def save_as_h5(dataset, h5file_path):
    with h5py.File(h5file_path, 'w') as dataset_file:
        for key in dataset.keys():
            dataset_file[key] = dataset[key]


if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)

    replay_samples = train(int(2e5))
    save_as_h5(replay_samples, "samples/particle-medium-replay-v0.hdf5")

    # random_samples, random_min, random_max = collect_offline_data(int(2e5), policy="random")
    # print(random_min, random_max)
    # save_as_h5(random_samples, "samples/particle-random-v0.hdf5")

    medium_samples, medium_min, medium_max = collect_offline_data(int(2e5), policy="medium")
    print(medium_min, medium_max)
    save_as_h5(medium_samples, "samples/particle-medium-v0.hdf5")
