import os
import shutil
import torch
import numpy as np
from utils.init_utils import get_algo, get_dataloader

def init_train(cfg, args):
    '''
        set device and seed
    '''
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    '''
        create directories
    '''
    results_dir = os.path.join('./results', cfg['dataset']['dataset'], cfg['train_id'])
    if os.path.isdir(results_dir):
        raise ValueError(f'{results_dir} Already existed!')
    
    os.makedirs(results_dir)
    shutil.copy('./configs/config.yaml', results_dir)

    '''
        get dataloader
    '''
    train_loader, val_loader = get_dataloader(cfg, args, 'trainval')
    '''
        get algorithm
    '''
    algo = get_algo(cfg)


    return algo, train_loader, val_loader, results_dir















