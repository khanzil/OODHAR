import os
from ruamel.yaml import YAML
import torch
import numpy as np
import shutil
import sys
from utils.init_utils import get_algo, get_domain_datasets, get_dataloaders, get_random_search_configs, get_random_seed

def init_loader(cfgs, args):
    '''
        Set device and seed. Note that RNG for data spliting and data loader are set in init_utils, these manual seed is mostly for init model weights and hyperparams.
    '''
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.search > 0:
        args.search = get_random_seed(args.seed,
                                      args.search,
                                      cfgs['algorithm'],
                                      cfgs['featurizer'])
    
        if args.cuda:
            torch.cuda.manual_seed_all(args.search)
        torch.manual_seed(args.search)
        np.random.seed(args.search)

        get_random_search_configs(cfgs)

    else:
        if args.cuda:
            torch.cuda.manual_seed_all(args.search)
        torch.manual_seed(args.search)
        np.random.seed(args.search)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    print(f"\nRNG state seed: {args.search}\n")

    '''
        create directories
    '''
    results_dir = os.path.join('./results', cfgs['dataset'], cfgs['train_id'])
    ckpts_dir = os.path.join('./ckpts', cfgs['dataset'], cfgs['train_id'])
    if cfgs['load_checkpoint'] == 'None':
        if os.path.isdir(results_dir):
            print(f"{results_dir} Already exist! Delete dir: Y/N?")
            ans = input()
            if ans != 'Y':
                raise ValueError(f"{results_dir} Already exist!")

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(ckpts_dir, exist_ok=True)

        yaml = YAML()
        with open(os.path.join(results_dir, f"config_{cfgs['train_id']}.yaml"), 'w') as f:
            yaml.dump(cfgs, f)

    '''
        get list of domain datasets
    '''    
    dom_datasets = get_domain_datasets(cfgs,args)
    # Dump cfgs to console
    yaml = YAML()
    yaml.dump(cfgs, sys.stdout)

    '''
        get dataloader
    '''
    loaders = get_dataloaders(cfgs, args, dom_datasets)


    return loaders, results_dir, ckpts_dir

def init_algo(cfgs, args):
    return get_algo(cfgs, args)
