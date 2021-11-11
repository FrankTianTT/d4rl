import numpy as np
from gym import utils
from d4rl.gym_mujoco import mujoco_env

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, noise_params=None):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4,
                                      qpos_begin=1,
                                      qvel_clip=True,
                                      add_cfrc_ext=False,
                                      noise_params=noise_params)
        utils.EzPickle.__init__(self, noise_params)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
