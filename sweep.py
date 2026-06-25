import argparse
from ruamel.yaml import YAML
import subprocess

if __name__ == '__main__':
    # cmd parser
    parser = argparse.ArgumentParser(description="Sweep for hyperparameter search",add_help=True)
    parser.add_argument('-c', '--config_file', help='Specify config file', metavar='FILE')
    parser.add_argument('--n_trials', type=int, default=3, help='Number of trials for each algo, affect how data is divided')
    parser.add_argument('--n_searchs', type=int, default=4, help='Number of hyperparameter searchs')
    parser.add_argument('--search_start', type=int, default=0, help='To do more search if needed')
    parser.add_argument('--algo', type=str)
    parser.add_argument('--featurizer', type=str)
    args = parser.parse_args()

    yaml = YAML()
    yaml.indent(mapping = 2, sequence=2, offset = 2)
    yaml.default_flow_style = False
    with open(args.config_file, 'r') as f:
        cfgs = yaml.load(f)

    # create a list of cfg to run each in a subprocess
    cfg_yaml_list = []
    
    for seed in range(args.n_trials):
        # only support single test domain for now, this seed controls RNG for dataset divison
        for search in range(1,args.n_searchs+1):
            cfg_yaml_list.append((f"./configs/sweep/config_seed{seed}_search{search}_{args.algo}_{args.featurizer}.yaml",seed,search))
            # create config_{i}.yaml for each cfg
            cfgs['train_id'] = f"seed{seed}_search{search}_{args.algo}_{args.featurizer}"
            cfgs['algorithm'] = args.algo
            cfgs['featurizer'] = args.featurizer
            with open(cfg_yaml_list[-1][0], 'w') as f:
                yaml.dump(cfgs, f)
            

    # # run subprocesses for each congis_{i}.yaml
    for i, (cfg_yaml,seed,search) in enumerate(cfg_yaml_list):
        if search < args.search_start:
            continue
        print(f'Starting {cfg_yaml}')
        subprocess.call(f'python train.py -c {cfg_yaml} train --num_workers=4 --seed={seed} --search={search}', shell=True)



















