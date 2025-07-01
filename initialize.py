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
    os.makedirs(os.path.join(results_dir, 'ckpts'))
    shutil.copy('./configs/config.yaml', results_dir)

    '''
        get dataloader
    '''
    train_loader, val_loader = get_dataloader(cfg, args, 'trainval')
    '''
        get algorithm
    '''
    algo = get_algo(cfg, args)


    return algo, train_loader, val_loader, results_dir

def init_test(cfg, args):
    results_dir = os.path.join('./results', cfg['dataset']['dataset'], cfg['train_id'])
    ckpts_dir = os.path.join(results_dir, 'ckpts')
    
    if cfg['test']['ckpt_dir'] == 'None':
        ckpt_file = os.listdir(ckpts_dir)[-1]
    else:
        ckpt_file = cfg['test']['ckpt']

    ckpt_path = os.path.join(ckpts_dir, ckpt_file)
    '''
        get dataloader
    '''
    test_loader = get_dataloader(cfg, args, 'test')
    '''
        get algorithm
    '''
    algo = get_algo(cfg, args)
    algo.load_ckpt(ckpt_path)

    return algo, test_loader, results_dir















