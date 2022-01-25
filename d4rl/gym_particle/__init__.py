from gym.envs.registration import register
from d4rl.gym_particle import particle
from d4rl import infos


for dataset in ['random', 'medium', 'medium-replay']:
    env_name = "particle-{}-v0".format(dataset)
    register(
        id=env_name,
        entry_point='d4rl.gym_particle.particle:get_particle_env',
        max_episode_steps=100,
        kwargs={
            'ref_min_score': infos.REF_MIN_SCORE[env_name],
            'ref_max_score': infos.REF_MAX_SCORE[env_name],
            'dataset_url': infos.DATASET_URLS[env_name],
        }
    )

for ratio in range(1, 11):
    env_name = 'particle-{}-v0'.format(ratio)
    register(
        id=env_name,
        entry_point='d4rl.gym_particle.particle:get_mixed_particle_env',
        max_episode_steps=100,
        kwargs={
            'ref_min_score': infos.REF_MIN_SCORE["particle-medium-v0"],
            'ref_max_score': infos.REF_MAX_SCORE["particle-medium-v0"],
            'dataset_urls': [infos.DATASET_URLS["particle-medium-v0"], infos.DATASET_URLS["particle-medium-replay-v0"]],
            "ratio": ratio
        }
    )