from collect_data import sample
from convert_buffer import convert
from rlkit_sac import train
import os

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "data")
os.makedirs(data_path, exist_ok=True)

def train_and_collect(env_name):
    env_data_path = os.path.join(data_path, env_name)
    os.makedirs(env_data_path, exist_ok=True)
    log_dir = train(env_name)
    convert(os.path.join(log_dir, "params.pkl"),
            os.path.join(env_data_path, "noisy-{}-medium-replay-v0.hdf5".format(env_name)))
    sample("noisy-{}-random-v0".format(env_name),
           os.path.join(log_dir, "params.pkl"),
           os.path.join(env_data_path, "noisy-{}-random-v0.hdf5".format(env_name)),
           1000,
           1000000,
           True)
    sample("noisy-{}-medium-v0".format(env_name),
           os.path.join(log_dir, "params.pkl"),
           os.path.join(env_data_path, "noisy-{}-medium-v0.hdf5".format(env_name)),
           1000,
           1000000,
           False)


if __name__ == '__main__':
    # train_and_collect("halfcheetah")
    train_and_collect("hopper")
