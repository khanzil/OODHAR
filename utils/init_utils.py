import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from utils.algo_utils import ERM, DANN
from utils.dataset_utils import Glasgow, GlasgowCollate

algos = {
    'ERM'   : ERM,
    'DANN'  : DANN
}

datasets = {
    'Glasgow'   : Glasgow
}

collate_fns = {
    'Glasgow'   : GlasgowCollate

}

def get_algo(cfg):
    return algos[cfg['algorithm']](cfg['model']['num_inputs'], cfg)

def get_dataloader(cfg, args, trainval_test):
    dataset_dir = cfg['dataset']['rootdir'] + '/' + cfg['dataset']['dataset']
    if trainval_test == 'trainval':
        if cfg['dataset']['val_fold'] == None: # split the train data if there's no val_fold, the ratio is 8 - 2
            folds = [fold for fold in os.listdir(dataset_dir) if os.path.isdir(fold) and fold not in fold not in cfg['dataset']['test_fold']]
            dataset = datasets[cfg['dataset']['dataset']](folds, cfg)
            idx = np.random.shuffle(np.arange(len(dataset)))
            train_dataset = Subset(dataset, idx[:int(0.8*len(dataset))+1])
            val_dataset = Subset(dataset, idx[-int(0.2*len(dataset)):])

            train_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=cfg['train']['batch_size'],
                                          shuffle=True,
                                          collate_fn=collate_fns['dataset']['dataset'],
                                          num_workers=args.num_workers
                                          )
            val_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=cfg['train']['batch_size'],
                                        shuffle=False,
                                        collate_fn=collate_fns['dataset']['dataset'],
                                        num_workers=args.num_workers
                                        )
        
        else:
            train_folds = [fold for fold in os.listdir(dataset_dir) if os.path.isdir(fold) and fold not in cfg['dataset']['val_fold'] and fold not in cfg['dataset']['test_fold']]
            train_dataset = datasets[cfg['dataset']['dataset']](train_folds, cfg)
            train_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=cfg['train']['batch_size'],
                                          shuffle=True,
                                          collate_fn=collate_fns['dataset']['dataset'],
                                          num_workers=args.num_workers
                                          )     
            val_folds = [fold for fold in cfg['dataset']['val_fold']]
            val_dataset = datasets[cfg['dataset']['dataset']](val_folds, cfg)
            val_dataloader = DataLoader(dataset=val_dataset, 
                                        batch_size=cfg['train']['batch_size'],
                                        shuffle=False,
                                        collate_fn=collate_fns['dataset']['dataset'],
                                        num_workers=args.num_workers
                                        )

        return train_dataloader, val_dataloader


    if trainval_test == 'test':
        folds = [fold for fold in cfg['dataset']['test_fold']]
        test_dataset = datasets[cfg['dataset']['dataset']](folds, cfg)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=cfg['test']['batch_size'],
                                     shuffle=False,
                                     collate_fn=collate_fns['dataset']['dataset'],
                                     num_workers=args.num_workers
                                     )

        return test_dataloader
    


































