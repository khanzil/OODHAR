import os
from ruamel.yaml import YAML
import torch
import numpy as np
from utils.init_utils import get_algo, get_dataloader, get_random_search_configs, get_random_seed

def init_train(cfgs, args):
    '''
        set device and seed
    '''
    cfgs['train_id'] = f'seed{args.seed}_search{args.search}_{args.algo}_{args.featurizer}'
    cfgs['algorithm'] = args.algo
    cfgs['featurizer'] = args.featurizer
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.search >= 0:
        args.search = get_random_seed(args.seed,
                                      args.search,
                                      args.algo,
                                      args.featurizer)
    
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


    '''
        create directories
    '''
    results_dir = os.path.join('./results', cfgs['dataset'], cfgs['train_id'])
    ckpts_dir = os.path.join('./ckpts', cfgs['dataset'], cfgs['train_id'])
    if cfgs['load_checkpoint'] == 'None':
        if os.path.isdir(results_dir):
            raise ValueError(f"{results_dir} Already existed!")

        os.makedirs(results_dir)
        os.makedirs(ckpts_dir)
        
        yaml = YAML()
        with open(os.path.join(results_dir, f"config_{cfgs['train_id']}.yaml"), 'w') as f:
            yaml.dump(cfgs, f)
        # shutil.copy('./configs/config.yaml', results_dir)

    '''
        get dataloader
    '''
    loaders = get_dataloader(cfgs, args)

    '''
        get algorithm
    '''
    algo = get_algo(cfgs, args)


    return algo, loaders, results_dir, ckpts_dir

def init_algo(cfgs, args):
    return get_algo(cfgs, args)
