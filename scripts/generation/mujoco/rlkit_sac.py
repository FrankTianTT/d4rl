import gym
from d4rl.gym_mujoco.ant import AntEnv
from d4rl.gym_mujoco.hopper import HopperEnv
from d4rl.gym_mujoco.half_cheetah import HalfCheetahEnv
from d4rl.gym_mujoco.walker2d import Walker2dEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

DEFAULT_NOISE_PARAMS = {"pos": {"xyz": 0.01,
                                "roll": 1,
                                "pitch": 1,
                                "yaw": 1,
                                "hinge": 1},
                        "vel": {"xyz": 0.01,
                                "rotate": 0.01,
                                "hinge": 0.01},
                        "action": 0.01}


class LogEnvReplayBuffer(EnvReplayBuffer):
    def __init__(self, max_replay_buffer_size, expl_env):
        env_info_sizes = {"terminal": 1,
                          "timeout": 1, }
        super(LogEnvReplayBuffer, self).__init__(max_replay_buffer_size, expl_env, env_info_sizes)

    def add_path(self, path):
        for i, (obs, action, reward, next_obs, terminal, env_info, done) in enumerate(zip(
                path["observations"], path["actions"], path["rewards"], path["next_observations"], path["terminals"],
                path["env_infos"], path["dones"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                env_info=env_info,
                done=done
            )
        self.terminate_episode()

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, done):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        self._env_infos["terminal"][self._top] = terminal
        self._env_infos["timeout"][self._top] = done and not terminal
        self._advance()

    def get_snapshot(self):
        env_infos = {}
        for key in self._env_info_keys:
            env_infos[key] = self._env_infos[key][:self._top]
        return dict(
            observations=self._observations[:self._top],
            next_observations=self._next_obs[:self._top],
            actions=self._actions[:self._top],
            rewards=self._rewards[:self._top],
            env_infos=env_infos,
        )


def experiment(variant, env_class):
    expl_env = NormalizedBoxEnv(env_class(DEFAULT_NOISE_PARAMS))
    eval_env = NormalizedBoxEnv(env_class(DEFAULT_NOISE_PARAMS))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = LogEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def train(env_name):
    env_classes = dict(hopper=HopperEnv,
                       halfcheetah=HalfCheetahEnv,
                       walker2d=Walker2dEnv)
    num_epochs = dict(hopper=200,
                      halfcheetah=100,
                      walker2d=100)

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=num_epochs[env_name],
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    log_dir = setup_logger("walker2d", variant=variant)
    ptu.set_gpu_mode(True)
    #
    experiment(variant, env_classes[env_name])
    #
    return log_dir
    #
    # return "/home/frank/Documents/project/rl/rlkit/data/walker2d/walker2d_2021_11_14_15_13_48_0000--s-0"

if __name__ == "__main__":
    train("walker2d")
