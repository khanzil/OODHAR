import os
import shutil
from ruamel.yaml import YAML
import torch
import numpy as np
from utils.init_utils import get_algo, get_dataloader

def init_train(cfgs, args):
    '''
        set device and seed
    '''
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    '''
        create directories
    '''
    results_dir = os.path.join('./results', cfgs['dataset'], cfgs['train_id'])
    if cfgs['load_checkpoint'] == 'None':
        if os.path.isdir(results_dir):
            raise ValueError(f"{results_dir} Already existed!")


        os.makedirs(os.path.join(results_dir, 'ckpts'))
        
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


    return algo, loaders, results_dir

def init_algo(cfgs, args):
    return get_algo(cfgs, args)




