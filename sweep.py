import argparse
import numpy as np
import itertools
from ruamel.yaml import YAML
from utils.sweep_utils import get_random_search_configs
import subprocess

if __name__ == '__main__':
    # cmd parser
    parser = argparse.ArgumentParser(description="Sweep for hyperparameter search",add_help=True)
    parser.add_argument('-c', '--config_file', help='Specify config file', metavar='FILE')
    parser.add_argument('--n_trials', type=int, default=3, help='Number of trials for each algo, affect how data is divided')
    parser.add_argument('--n_searchs', type=int, default=20, help='Number of hyperparameter searchs')
    parser.add_argument('--trial_seed', type=int, default=0, help='Seed for sweeping, affect how hyperparameter is generated')
    parser.add_argument('--n_test_doms', type=int, default=1, help='Number of test domains')
    parser.add_argument('--algo', type=str)
    parser.add_argument('--featurizer', type=str)
    args = parser.parse_args()

    yaml = YAML()
    yaml.indent(mapping = 2, sequence=2, offset = 2)
    yaml.default_flow_style = False
    with open(args.config_file, 'r') as f:
        cfg = yaml.load(f)

    # create a list of cfg to run each in a subprocess
    np.random.seed(args.trial_seed)
    cfg_yaml_list = []
    train_id = 0
    
    for seed in range(args.n_trials):
        # only support single test domain for now
        for search in range(args.n_searchs):
            new_cfg = get_random_search_configs(cfg, seed, search, args.algo, args.featurizer)
            cfg_yaml_list.append(f'./configs/sweep/config_{new_cfg['train_id']}.yaml')
            # create config_{i}.yaml for each cfg
            with open(cfg_yaml_list[-1], 'w') as f:
                yaml.dump(new_cfg, f)

            train_id += 1
            

    # # run subprocesses for each congis_{i}.yaml
    for seed, cfg_yaml in enumerate(cfg_yaml_list):
        print(f'Starting {cfg_yaml}')
        subprocess.call(f'python train.py -c {cfg_yaml} train --num_workers=4 --seed={seed}', shell=True)



















