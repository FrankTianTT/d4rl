from d4rl import offline_env
from d4rl.gym_mujoco.ant import AntEnv
from d4rl.gym_mujoco.hopper import HopperEnv
from d4rl.gym_mujoco.half_cheetah import HalfCheetahEnv
from d4rl.gym_mujoco.walker2d import Walker2dEnv
from d4rl.utils.wrappers import NormalizedBoxEnv

DEFAULT_NOISE_PARAMS = {"pos": {"xyz": 0.01,
                                "roll": 1,
                                "pitch": 1,
                                "yaw": 1,
                                "hinge": 1},
                        "vel": {"xyz": 0.01,
                                "rotate": 0.01,
                                "hinge": 0.01},
                        "action": 0.01}

class OfflineAntEnv(AntEnv, offline_env.OfflineEnv):
    def __init__(self, noise_params=None, **kwargs):
        AntEnv.__init__(self, noise_params)
        offline_env.OfflineEnv.__init__(self, **kwargs)


class OfflineHopperEnv(HopperEnv, offline_env.OfflineEnv):
    def __init__(self, noise_params=None, **kwargs):
        HopperEnv.__init__(self, noise_params)
        offline_env.OfflineEnv.__init__(self, **kwargs)


class OfflineHalfCheetahEnv(HalfCheetahEnv, offline_env.OfflineEnv):
    def __init__(self, noise_params=None, **kwargs):
        HalfCheetahEnv.__init__(self, noise_params)
        offline_env.OfflineEnv.__init__(self, **kwargs)


class OfflineWalker2dEnv(Walker2dEnv, offline_env.OfflineEnv):
    def __init__(self, noise_params=None, **kwargs):
        Walker2dEnv.__init__(self, noise_params)
        offline_env.OfflineEnv.__init__(self, **kwargs)


def get_ant_env(**kwargs):
    return NormalizedBoxEnv(OfflineAntEnv(**kwargs))


def get_cheetah_env(**kwargs):
    return NormalizedBoxEnv(OfflineHalfCheetahEnv(**kwargs))


def get_hopper_env(**kwargs):
    return NormalizedBoxEnv(OfflineHopperEnv(**kwargs))


def get_walker_env(**kwargs):
    return NormalizedBoxEnv(OfflineWalker2dEnv(**kwargs))


def get_noisy_ant_env(**kwargs):
    return NormalizedBoxEnv(OfflineAntEnv(DEFAULT_NOISE_PARAMS, **kwargs))


def get_noisy_cheetah_env(**kwargs):
    return NormalizedBoxEnv(OfflineHalfCheetahEnv(DEFAULT_NOISE_PARAMS, **kwargs))


def get_noisy_hopper_env(**kwargs):
    return NormalizedBoxEnv(OfflineHopperEnv(DEFAULT_NOISE_PARAMS, **kwargs))


def get_noisy_walker_env(**kwargs):
    return NormalizedBoxEnv(OfflineWalker2dEnv(DEFAULT_NOISE_PARAMS, **kwargs))

class MixedOfflineHopperEnv(Walker2dEnv, offline_env.MixedOfflineEnv):
    def __init__(self, noise_params=None, **kwargs):
        Walker2dEnv.__init__(self, noise_params)
        offline_env.MixedOfflineEnv.__init__(self, **kwargs)

def get_mixed_hopper_env(**kwargs):
    return NormalizedBoxEnv(MixedOfflineHopperEnv(**kwargs))


if __name__ == '__main__':
    env = get_noisy_ant_env()
    env.reset()
    env.render()

    while True:
        o, r, d, i = env.step(env.action_space.sample())
        env.render()

        if d:
            env.reset()
