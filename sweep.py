import argparse
from ruamel.yaml import YAML
import subprocess
import os
import torch
import time
import threading
import sys

def stream(proc, gpu_id):
    for line in iter(proc.stdout.readline, b''):
        with lock:                          # one write at a time
            sys.stdout.write(f"[GPU {gpu_id}] {line.decode()}")
            sys.stdout.flush()

if __name__ == '__main__':
    # cmd parser
    parser = argparse.ArgumentParser(description="Sweep for hyperparameter search",add_help=True)
    parser.add_argument('-c', '--config_file', help='Specify config file', metavar='FILE')
    parser.add_argument('--n_trials', type=int, default=3, help='Number of trials for each algo, affect how data is divided')
    parser.add_argument('--trial_start', type=int, default=0, help='To do more trial if needed')
    parser.add_argument('--n_searchs', type=int, default=4, help='Number of hyperparameter searchs')
    parser.add_argument('--search_start', type=int, default=1, help='To do more search if needed')
    parser.add_argument('--single_gpu', type=bool, default=True, help='Set to False to use more than 1 GPU')
    parser.add_argument('--algo', type=str)
    parser.add_argument('--featurizer', type=str)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()

    yaml = YAML()
    yaml.indent(mapping = 2, sequence=2, offset = 2)
    yaml.default_flow_style = False
    with open(args.config_file, 'r') as f:
        cfgs = yaml.load(f)

    # create a list of cfg to run each in a subprocess
    cfg_yaml_list = []
    
    for seed in range(args.trial_start, args.n_trials):
        # only support single test domain for now, this seed controls RNG for dataset divison
        for search in range(args.search_start,args.n_searchs+1):
            train_cfg_dir = f"./configs/sweep/config_seed{seed}_search{search}_{args.algo}_{args.featurizer}.yaml"
            # create config_{i}.yaml for each cfg
            cfgs['train_id'] = f"seed{seed}_search{search}_{args.algo}_{args.featurizer}"
            cfgs['algorithm'] = args.algo
            cfgs['featurizer'] = args.featurizer
            with open(train_cfg_dir, 'w') as f:
                yaml.dump(cfgs, f)
            cfg_yaml_list.append((f"./configs/sweep/config_seed{seed}_search{search}_{args.algo}_{args.featurizer}.yaml",seed,search))

            # print(f"Starting {cfgs['train_id']}")
            # subprocess.call(f"python train.py -c {train_cfg_dir} train --num_workers={args.num_workers} --seed={seed} --search={search}", shell=True)
    print(args.single_gpu)
    print(type(args.single_gpu))
    # # run subprocesses for each congis_{i}.yaml
    if args.single_gpu:
        for i, (cfg_yaml,seed,search) in enumerate(cfg_yaml_list):
            print(f'Starting {cfg_yaml}')
            subprocess.call(f'python train.py -c {cfg_yaml} train --num_workers={args.num_workers} --seed={seed} --search={search}', shell=True)
    else:
        print('Starting multi-GPU training')
        try:
            # Get list of GPUs from env, split by ',' and remove empty string ''
            # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
            available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
        except Exception:
            # If the env variable is not set, we use all GPUs
            available_gpus = [str(x) for x in range(torch.cuda.device_count())]
        n_gpus = len(available_gpus)
        procs_by_gpu  = [None] * n_gpus
        threads_by_gpu = [None] * n_gpus
        lock = threading.Lock()

        while len(cfg_yaml_list) > 0:
            for idx, gpu_idx in enumerate(available_gpus):
                proc = procs_by_gpu[idx]
                if (proc is None) or (proc.poll() is not None):
                    cfg_yaml,seed,search = cfg_yaml_list.pop(0)
                    print(f'Starting {cfg_yaml}')
                    new_proc = subprocess.Popen(
                        f'CUDA_VISIBLE_DEVICES={gpu_idx} python train.py -c {cfg_yaml} train --num_workers={args.num_workers} --seed={seed} --search={search}',
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    procs_by_gpu[idx] = new_proc
                    t = threading.Thread(target=stream, args=(new_proc, gpu_idx), daemon=True,)
                    t.start()
                    threads_by_gpu[idx] = t
                    break
            time.sleep(1)

        # Wait for remaining processes and their output streams
        for t in threads_by_gpu:
            if t is not None:
                t.join()
        for p in procs_by_gpu:
            if p is not None:
                p.wait()















