from gym.envs.registration import register
from d4rl.gym_mujoco import gym_envs
from d4rl import infos

# V1 envs
for agent in ['hopper', 'halfcheetah', 'ant', 'walker2d']:
    for dataset in ['random', 'medium', 'expert', 'medium-expert', 'medium-replay', 'full-replay']:
        for version in ['v1', 'v2']:
            env_name = '%s-%s-%s' % (agent, dataset, version)
            register(
                id=env_name,
                entry_point='d4rl.gym_mujoco.gym_envs:get_%s_env' % agent.replace('halfcheetah', 'cheetah').replace(
                    'walker2d', 'walker'),
                max_episode_steps=1000,
                kwargs={
                    'ref_min_score': infos.REF_MIN_SCORE[env_name],
                    'ref_max_score': infos.REF_MAX_SCORE[env_name],
                    'dataset_url': infos.DATASET_URLS[env_name]
                }
            )

score = {
    "random": {
        "hopper": -20.272305,
        "halfcheetah": -280.178953,
        "ant": -325.6,
        "walker2d": 1.629008},
    "expert":
        {"hopper": 3234.3,
         "halfcheetah": 12135.0,
         "ant": 3879.7,
         "walker2d": 4592.3}
}
for agent in ['hopper', 'halfcheetah', 'ant', 'walker2d']:
    for dataset in ['random', 'medium', 'medium-replay']:
        env_name = '%s-%s-v0' % (agent, dataset)
        register(
            id=env_name,
            entry_point='d4rl.gym_mujoco.gym_envs:get_%s_env' % agent.replace('halfcheetah', 'cheetah').replace(
                'walker2d', 'walker'),
            max_episode_steps=1000,
            kwargs={
                'ref_min_score': score["random"][agent],
                'ref_max_score': score["expert"][agent],
                'dataset_url': "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/{}.hdf5".format(env_name)
            }
        )


noisy_score = {
    "random": {
        "hopper": 0,
        "halfcheetah": 0,
        "ant": 0,
        "walker2d": 0},
    "expert":
        {"hopper": 0,
         "halfcheetah": 0,
         "ant": 0,
         "walker2d": 0}
}

for agent in ['hopper', 'halfcheetah', 'ant', 'walker2d']:
    for dataset in ['random', 'medium', 'medium-replay']:
        env_name = 'noisy-%s-%s-v0' % (agent, dataset)
        register(
            id=env_name,
            entry_point='d4rl.gym_mujoco.gym_envs:get_noisy_%s_env' % agent.replace('halfcheetah', 'cheetah').replace(
                'walker2d', 'walker'),
            max_episode_steps=1000,
            kwargs={
                'ref_min_score': noisy_score["random"][agent],
                'ref_max_score': noisy_score["expert"][agent],
                'dataset_url': "http://drive.franktian.top/offline_dataset/{}.hdf5".format(env_name)
            }
        )
for ratio in range(10):
    print(ratio)
register(
    id=env_name,
    entry_point='d4rl.gym_mujoco.gym_envs:get_noisy_%s_env' % agent.replace('halfcheetah', 'cheetah').replace(
        'walker2d', 'walker'),
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': noisy_score["random"][agent],
        'ref_max_score': noisy_score["expert"][agent],
        'dataset_url': "http://drive.franktian.top/offline_dataset/{}.hdf5".format(env_name)
    }
)