import os
import sys
import collections
import numpy as np
import random
import d4rl.infos
from d4rl.offline_env import set_dataset_path, get_keys

SUPPRESS_MESSAGES = bool(os.environ.get('D4RL_SUPPRESS_IMPORT_ERROR', 0))

_ERROR_MESSAGE = 'Warning: %s failed to import. Set the environment variable D4RL_SUPPRESS_IMPORT_ERROR=1 to suppress this message.'

try:
    import d4rl.locomotion
    import d4rl.hand_manipulation_suite
    import d4rl.pointmaze
    import d4rl.gym_minigrid
    import d4rl.gym_mujoco
except ImportError as e:
    if not SUPPRESS_MESSAGES:
        print(_ERROR_MESSAGE % 'Mujoco-based envs', file=sys.stderr)
        print(e, file=sys.stderr)

try:
    import d4rl.flow
except ImportError as e:
    if not SUPPRESS_MESSAGES:
        print(_ERROR_MESSAGE % 'Flow', file=sys.stderr)
        print(e, file=sys.stderr)

try:
    import d4rl.kitchen
except ImportError as e:
    if not SUPPRESS_MESSAGES:
        print(_ERROR_MESSAGE % 'FrankaKitchen', file=sys.stderr)
        print(e, file=sys.stderr)

try:
    import d4rl.carla
except ImportError as e:
    if not SUPPRESS_MESSAGES:
        print(_ERROR_MESSAGE % 'CARLA', file=sys.stderr)
        print(e, file=sys.stderr)
        
try:
    import d4rl.gym_bullet
    import d4rl.pointmaze_bullet
except ImportError as e:
    if not SUPPRESS_MESSAGES:
        print(_ERROR_MESSAGE % 'GymBullet', file=sys.stderr)
        print(e, file=sys.stderr)

import d4rl.gym_particle

def reverse_normalized_score(env_name, score):
    ref_min_score = d4rl.infos.REF_MIN_SCORE[env_name]
    ref_max_score = d4rl.infos.REF_MAX_SCORE[env_name]
    return (score * (ref_max_score - ref_min_score)) + ref_min_score

def get_normalized_score(env_name, score):
    ref_min_score = d4rl.infos.REF_MIN_SCORE[env_name]
    ref_max_score = d4rl.infos.REF_MAX_SCORE[env_name]
    return (score - ref_min_score) / (ref_max_score - ref_min_score)

def qlearning_dataset(env):
    if hasattr(env, "ratio"):
        dataset1, dataset2 = env.get_dataset()
        qlearning_dataset1 = get_qlearning_dataset(dataset1, ratio=10)
        qlearning_dataset2 = get_qlearning_dataset(dataset2, env.ratio)
        for key in ['observations', 'actions', 'rewards', 'terminals', 'next_observations']:
            qlearning_dataset1[key] = np.concatenate([qlearning_dataset1[key], qlearning_dataset2[key]])
        return qlearning_dataset1

    else:
        return get_qlearning_dataset(env.get_dataset(), ratio=10)

def get_qlearning_dataset(dataset, terminate_on_end=False, ratio=1, **kwargs):
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        final_timestep = dataset['timeouts'][i]

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0

        if random.randint(0, 9) < ratio:
            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def sequence_dataset(env, dataset=None, **kwargs):
    """
    Returns an iterator through trajectories.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1

