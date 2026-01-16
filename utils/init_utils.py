import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from utils.algo_utils import ERM, DANN
from utils.dataset_utils import Glasgow, GlasgowCollate

algos_dict = {
    'ERM'   : ERM,
    'DANN'  : DANN
}

datasets_dict = {
    'Glasgow'   : Glasgow
}

collate_fns_dict = {
    'Glasgow'   : GlasgowCollate
}

def get_algo(cfg, args):
    algo = algos_dict[cfg['algorithm']](cfg, args)
    if not args.no_cuda:
        algo.cuda()
    return algo

def get_dataloader(cfg, args, trainval_test):
    dataset_dir = os.path.join(cfg['dataset']['rootdir'], cfg['dataset']['dataset']) # directory contains all data folders
    val_folds = [fold.strip() for fold in cfg['dataset']['val_fold'].split(',')] # name of val_folders
    test_folds = [fold.strip() for fold in cfg['dataset']['test_fold'].split(',')] # name of test_folders
    train_folds = [fold for fold in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,fold)) and fold not in val_folds and fold not in test_folds]

    if trainval_test == 'trainval':
        if cfg['dataset']['val_fold'] == 'None': # split the train data if there's no val_fold, the ratio is 8 - 2
            dataset = datasets_dict[cfg['dataset']['dataset']](train_folds, cfg)
            idx = np.arange(len(dataset))
            np.random.shuffle(idx)
            train_dataset = Subset(dataset, idx[:int(0.8*len(dataset))+1])
            val_dataset = Subset(dataset, idx[int(0.8*len(dataset))+1:])

            train_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=cfg['train']['batch_size'],
                                          shuffle=True,
                                          collate_fn=collate_fns_dict[cfg['dataset']['dataset']],
                                          num_workers=args.num_workers
                                          )
            val_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=cfg['train']['batch_size'],
                                        shuffle=False,
                                        collate_fn=collate_fns_dict[cfg['dataset']['dataset']],
                                        num_workers=args.num_workers
                                        )
        
        else: 
            if val_folds not in os.listdir(dataset_dir):
                raise ValueError(f'Val folder {val_folds} not existed')
            train_dataset = datasets_dict[cfg['dataset']['dataset']](train_folds, cfg)
            train_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=cfg['train']['batch_size'],
                                          shuffle=True,
                                          collate_fn=collate_fns_dict[cfg['dataset']['dataset']],
                                          num_workers=args.num_workers
                                          )     
            val_dataset = datasets_dict[cfg['dataset']['dataset']](val_folds, cfg)
            val_dataloader = DataLoader(dataset=val_dataset, 
                                        batch_size=cfg['train']['batch_size'],
                                        shuffle=False,
                                        collate_fn=collate_fns_dict[cfg['dataset']['dataset']],
                                        num_workers=args.num_workers
                                        )

        return train_dataloader, val_dataloader

    elif trainval_test == 'test':
        if test_folds not in os.listdir(dataset_dir):
            raise ValueError(f'Test folder {test_folds} not existed')
        test_dataset = datasets_dict[cfg['dataset']['dataset']](test_folds, cfg)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=cfg['test']['batch_size'],
                                     shuffle=False,
                                     collate_fn=collate_fns_dict[cfg['dataset']['dataset']],
                                     num_workers=args.num_workers
                                     )

        return test_dataloader
    


































