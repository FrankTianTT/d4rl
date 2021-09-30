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
from collections import defaultdict, Counter
import h5py
from tqdm import tqdm

max_episode_steps = 100


def train(total_timesteps=int(5e4)):
    eval_env = TimeLimit(ParticleEnv(), max_episode_steps)
    env = TimeLimit(ParticleEnv(), max_episode_steps)

    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                                 log_path='./logs/results', eval_freq=max_episode_steps)

    checkpoint_callback = CheckpointCallback(save_freq=max_episode_steps, save_path='./logs/')
    callback = CallbackList([checkpoint_callback, eval_callback])

    model = SAC('MlpPolicy', env, tensorboard_log="./log",
                batch_size=max_episode_steps)

    model.learn(total_timesteps, callback=callback)
    # return collect_offline_data_from_model(model)


def collect_offline_data_from_model(model):
    replay_buffer = model.replay_buffer
    unscale_action = model.policy.unscale_action
    pos = replay_buffer.pos
    # SAC of sb3 will scale action automatically, so un-scale it manually.
    samples = {'observations': replay_buffer.observations[:pos].reshape(pos, -1),
               'actions': unscale_action(replay_buffer.actions[:pos].reshape(pos, -1)),
               'rewards': replay_buffer.rewards[:pos].reshape(pos),
               'terminals': replay_buffer.dones[:pos].reshape(pos),
               'timeouts': replay_buffer.timeouts[:pos].reshape(pos)}

    return samples


def collect_offline_data(num=int(2e5), policy_path=None):
    env = TimeLimit(ParticleEnv(), max_episode_steps)
    episode_rewards = []

    t = 0
    episode_reward = 0
    obs = env.reset()
    samples = defaultdict(list)
    model = None
    if policy_path is not None:
        model = SAC.load(policy_path)

    while t < num:
        if model is None:
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


def collect_multi_offline_data(num=int(2e5), policy_path_dir=None):
    policy_paths = os.listdir(policy_path_dir)
    policy_paths = list(filter(lambda x: x.endswith("zip"), policy_paths))
    num_pre_policy = int((num / len(policy_paths) / max_episode_steps) + 1) * max_episode_steps

    episode_rewards = []
    samples = defaultdict(list)
    for policy_name in tqdm(policy_paths):
        path = os.path.join(policy_path_dir, policy_name)
        policy_samples, policy_min, policy_max = collect_offline_data(num_pre_policy, path)
        episode_rewards.extend([policy_min, policy_max])
        for key in policy_samples.keys():
            samples[key].append(policy_samples[key])

    np_samples = {}
    for key in samples.keys():
        np_samples[key] = np.concatenate(samples[key])

    return np_samples, min(episode_rewards), max(episode_rewards)


def save_as_h5(dataset, h5file_path):
    with h5py.File(h5file_path, 'w') as dataset_file:
        for key in dataset.keys():
            dataset_file[key] = dataset[key]

#
# def visualize_data(sample_path):
#     import pandas as pd
#     from pandas_profiling import ProfileReport
#     obs_dim, act_dim, inputs, outputs = load_env(env_name, use_diff_predict=False, normalize=False)
#     df = pd.DataFrame(np.concatenate([inputs, outputs], axis=1))
#     columns = ["s_t_{}".format(i + 1) for i in range(inputs.shape[1] - 1)] + ["action", "reward"] + \
#               ["s_t'_{}".format(i + 1) for i in range(inputs.shape[1] - 1)]
#     df.columns = columns
#
#     profile = ProfileReport(df, title="{} Report".format(env_name))
#     profile.to_file("{}_report.html".format(env_name))


if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)

    train(int(1e4))
    replay_samples, replay_min, replay_max = collect_multi_offline_data(int(2e5), policy_path_dir="./logs")
    save_as_h5(replay_samples, "samples/particle-medium-replay-v0.hdf5")

    random_samples, random_min, random_max = collect_offline_data(int(1e6))
    save_as_h5(random_samples, "samples/particle-random-v0.hdf5")

    medium_samples, medium_min, medium_max = collect_offline_data(int(1e6), policy_path='./logs/best_model/best_model.zip')
    save_as_h5(medium_samples, "samples/particle-medium-v0.hdf5")
