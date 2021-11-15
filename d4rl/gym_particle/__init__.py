from gym.envs.registration import register
from d4rl.gym_particle import particle



DATASET_URLS = {
    'particle-random-v0': 'http://drive.franktian.top/offline_dataset/particle-random-v0.hdf5',
    'particle-medium-v0': 'http://drive.franktian.top/offline_dataset/particle-medium-v0.hdf5',
    'particle-medium-replay-v0': 'http://drive.franktian.top/offline_dataset/particle-medium-replay-v0.hdf5'
}

sac_min = -160.7
sac_max = 261.5

for dataset in ['random', 'medium', 'medium-replay']:
    env_name = "particle-{}-v0".format(dataset)
    register(
        id=env_name,
        entry_point='d4rl.gym_particle.particle:get_particle_env',
        max_episode_steps=300,
        kwargs={
            'ref_min_score': sac_min,
            'ref_max_score': sac_max,
            'dataset_url': DATASET_URLS[env_name]
        }
    )
