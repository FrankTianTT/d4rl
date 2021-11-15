import argparse
import re

import h5py
import torch
import numpy as np

itr_re = re.compile(r'itr_(?P<itr>[0-9]+).pkl')

def load(pklfile):
    params = torch.load(pklfile)
    env_infos = params['replay_buffer/env_infos']
    results = { 
        'observations': params['replay_buffer/observations'],
        'next_observations': params['replay_buffer/next_observations'],
        'actions': params['replay_buffer/actions'],
        'rewards': params['replay_buffer/rewards'],
        'terminals': env_infos['terminal'].squeeze(),
        'timeouts': env_infos['timeout'].squeeze(),
    }
    return results

def get_pkl_itr(pklfile):
    match = itr_re.search(pklfile)
    if match:
        return match.group('itr')
    raise ValueError(pklfile+" has no iteration number.")

def convert(pklfile, output_file):
    data = load(pklfile)
    hfile = h5py.File(output_file, 'w')
    for k in data:
        hfile.create_dataset(k, data=data[k], compression='gzip')

    hfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pklfile', type=str)
    parser.add_argument('--output_file', type=str, default='output.hdf5')
    args = parser.parse_args()

    convert(args.pklfile, args.output_file)
