from gym.envs.registration import register
from d4rl.gym_particle import particle

DATASET_URLS = {
    'particle-random-v0': 'http://114.212.21.162:6868/offline_dataset/particle-random-v0.hdf5',
    'particle-medium-v0': 'http://114.212.21.162:6868/offline_dataset/particle-medium-v0.hdf5',
    'particle-medium-replay-v0': 'http://114.212.21.162:6868/offline_dataset/particle-medium-replay-v0.hdf5',
    'particle-full-v0': 'http://114.212.21.162:6868/offline_dataset/particle-full-v0.hdf5',
    'particle-full-replay-v0': 'http://114.212.21.162:6868/offline_dataset/particle-full-replay-v0.hdf5',
}

REF_MIN_SCORE = {'particle-random-v0': -248.32,
                 'particle-medium-v0': 0,
                 'particle-medium-replay-v0': 0,
                 'particle-full-v0': 180.79,
                 'particle-full-replay-v0': 25.07}
REF_MAX_SCORE = {'particle-random-v0': 169.90,
                 'particle-medium-v0': 0,
                 'particle-medium-replay-v0': 0,
                 'particle-full-v0': 208.60,
                 'particle-full-replay-v0': 193.9}

for dataset in ['random', 'medium', 'medium-replay', 'full', 'full-replay']:
    env_name = "particle-{}-v0".format(dataset)
    register(
        id=env_name,
        entry_point='d4rl.gym_particle.particle:get_particle_env',
        max_episode_steps=300,
        kwargs={
            'ref_min_score': REF_MIN_SCORE[env_name],
            'ref_max_score': REF_MAX_SCORE[env_name],
            'dataset_url': DATASET_URLS[env_name]
        }
    )
