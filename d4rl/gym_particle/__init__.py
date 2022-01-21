from gym.envs.registration import register
from d4rl.gym_particle import particle



DATASET_URLS = {
    'particle-random-v0': 'https://box.nju.edu.cn/f/f988a549016646928b1a/?dl=1',
    'particle-medium-v0': 'https://box.nju.edu.cn/f/8352da217e6e4614adfa/?dl=1',
    'particle-medium-replay-v0': 'https://box.nju.edu.cn/f/af872f1d1ef84d1189a4/?dl=1'
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
