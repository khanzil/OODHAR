import os
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
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
    return algos_dict[cfg['algorithm']](cfg, args)

def get_dataloader(cfg, args, trainval_test):
    dataset_dir = os.path.join(cfg['rootdir'], cfg['dataset']) # directory contains all data folders
    dom_list = [fold for fold in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,fold))]
    loaders = []
   
    if cfg['test_dom'] == 'None':
        # This code is using train-valication domain split and sweep through all test domain combination
        for i_test_dom, test_dom in enumerate(dom_list):
            train_dataset = []
            val_dataset = []

            for fold in dom_list:
                dataset = datasets_dict[cfg['dataset']](fold, cfg)
                if fold == test_dom:
                    test_loaders = DataLoader(dataset=dataset,
                                     batch_size=cfg['batch_size'],
                                     shuffle=False,
                                     collate_fn=collate_fns_dict[cfg['dataset']],
                                     num_workers=args.num_workers
                                     )

                else:
                    idx = np.arange(len(dataset))
                    np.random.shuffle(idx)
                    train_dataset.append(Subset(dataset, idx[:int(0.8*len(dataset))+1]))
                    val_dataset.append(Subset(dataset, idx[int(0.8*len(dataset))+1:]))
            
            train_dataset = ConcatDataset(train_dataset)
            val_dataset = ConcatDataset(val_dataset)

            train_loaders = DataLoader(dataset=train_dataset,
                                        batch_size=cfg['batch_size'],
                                        shuffle=True,
                                        collate_fn=collate_fns_dict[cfg['dataset']],
                                        num_workers=args.num_workers
                                        )
            
            val_loaders = DataLoader(dataset=val_dataset,
                                    batch_size=cfg['batch_size'],
                                    shuffle=False,
                                    collate_fn=collate_fns_dict[cfg['dataset']],
                                    num_workers=args.num_workers
                                    )
            
            loaders.append((train_loaders, val_loaders, test_loaders))
    else:
        raise NotImplementedError('Only support all enrivonment be as test for now')
    
    return loaders
    
    
    
    
    
    
    
    # val_doms = [fold.strip() for fold in cfg['val_dom'].split(',')] # name of val_dom
    # test_doms = [fold.strip() for fold in cfg['test_dom'].split(',')] # name of test_dom
    
    # train_folds = [fold for fold in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,fold)) and fold not in val_doms and fold not in test_doms]

    # if trainval_test == 'trainval':
    #     if cfg['val_dom'] == 'None': # split the train data if there's no val_dom, the ratio is 8 - 2
    #         dataset = datasets_dict[cfg['dataset']](train_folds, cfg)
    #         idx = np.arange(len(dataset))
    #         np.random.shuffle(idx)
            
    #         train_dataset = Subset(dataset, idx[:int(0.8*len(dataset))+1])
    #         val_dataset = Subset(dataset, idx[int(0.8*len(dataset))+1:])

    #         train_loaders = DataLoader(dataset=train_dataset,
    #                                       batch_size=cfg['batch_size'],
    #                                       shuffle=True,
    #                                       collate_fn=collate_fns_dict[cfg['dataset']],
    #                                       num_workers=args.num_workers
    #                                       )
    #         val_loaders = DataLoader(dataset=val_dataset,
    #                                     batch_size=cfg['batch_size'],
    #                                     shuffle=False,
    #                                     collate_fn=collate_fns_dict[cfg['dataset']],
    #                                     num_workers=args.num_workers
    #                                     )
        
    #     else: 
    #         for fold in val_doms:
    #             if fold not in os.listdir(dataset_dir):
    #                 raise ValueError(f'Val folders {val_doms} not existed in {dataset_dir}')
            
    #         train_dataset = datasets_dict[cfg['dataset']](train_folds, cfg)
    #         train_loaders = DataLoader(dataset=train_dataset,
    #                                       batch_size=cfg['batch_size'],
    #                                       shuffle=True,
    #                                       collate_fn=collate_fns_dict[cfg['dataset']],
    #                                       num_workers=args.num_workers
    #                                       )     
    #         val_dataset = datasets_dict[cfg['dataset']](val_doms, cfg)
    #         val_loaders = DataLoader(dataset=val_dataset, 
    #                                     batch_size=cfg['batch_size'],
    #                                     shuffle=False,
    #                                     collate_fn=collate_fns_dict[cfg['dataset']],
    #                                     num_workers=args.num_workers
    #                                     )

    #     return train_loaders, val_loaders

    # elif trainval_test == 'test':
    #     for fold in test_doms:
    #         if fold not in os.listdir(dataset_dir):
    #             raise ValueError(f'Test folder {fold} not existed')
            
    #     test_dataset = datasets_dict[cfg['dataset']](test_doms, cfg)
    #     test_loaders = DataLoader(dataset=test_dataset,
    #                                  batch_size=cfg['batch_size'],
    #                                  shuffle=False,
    #                                  collate_fn=collate_fns_dict[cfg['dataset']],
    #                                  num_workers=args.num_workers
    #                                  )

    #     return test_loaders
    


































