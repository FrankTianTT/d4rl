import gym
import d4rl
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def from_path(path):
    dataset = {}
    h5path = "/home/frank/Documents/project/offline_rl/d4rl/output.hdf5"
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            dataset[k] = dataset_file[k][:]

    return dataset

def from_name(env_name):
    env = gym.make(env_name)

    dataset = env.get_dataset()

    return dataset

if __name__ == '__main__':
    dataset = from_name('noisy-halfcheetah-random-v0')

    rewards = dataset["rewards"]
    terminals = dataset["terminals"]
    timeouts = dataset["timeouts"]
    print("sample num:", len(terminals))

    dones = map(lambda x: bool(x[0]) or bool(x[1]), zip(terminals, timeouts))

    returns = []
    this_return = 0
    for r, d in zip(rewards, dones):
        if not d:
            this_return += r
        else:
            returns.append(this_return)
            this_return = 0

    print(sum(returns) / len(returns))
    print("rollout num:", len(returns))
    plt.plot(range(len(returns)), returns)
    plt.show()