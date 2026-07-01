import os
import numpy as np
import hashlib
from torch.utils.data import DataLoader, random_split, ConcatDataset

from utils.algo_utils import *
from utils.dataset_utils import *
from utils.sweep_utils import get_random_search_configs

algos_dict = {
    'ERM'   : ERM,
    'DANN'  : DANN,
    'IRM'   : IRM,
    'Fish'  : Fish,
    'VRex'  : VRex,
}

datasets_dict = {
    'Glasgow'   : Glasgow
}

def get_random_seed(*args):
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**32)

def get_algo(cfgs, args):
    return algos_dict[cfgs['algorithm']](cfgs, args)

def get_dataloader(cfgs, args):
    dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset']) # directory contains all data folders
    dom_list = [fold for fold in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir,fold))]
    loaders = []
    dataset_num_workers = -1
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    if cfgs['test_dom'] == 'None':
        # This code is using train-valication domain split and sweep through all test domain combination
        for i_test_dom, test_dom in enumerate(dom_list):
            train_datasets = []
            val_datasets = []
            test_datasets = []

            for fold in dom_list:
                dataset = datasets_dict[cfgs['dataset']](fold, cfgs)
                if dataset_num_workers == -1:
                    dataset_num_workers = dataset.num_workers
                if fold == test_dom:
                    test_datasets.append(dataset)

                else:
                    train_dataset, val_dataset = random_split(dataset, 
                                                              [cfgs['train_split'], 
                                                               1-cfgs['train_split']],
                                                              generator=generator)

                    train_weights = make_weights_for_balanced_classes(train_dataset)
                    train_datasets.append((train_dataset, train_weights))

                    val_datasets.append(val_dataset)

            total_batch_size = len(train_datasets)*cfgs['batch_size']

            
            train_loader = [InfiniteDataLoader(dataset=dataset,
                                                weights=weights,
                                                batch_size=cfgs['batch_size'],
                                                num_workers=dataset_num_workers)
                             for dataset, weights in train_datasets]
            train_loader = zip(*train_loader)

            in_val_loader = DataLoader(dataset=ConcatDataset([dataset for dataset, _ in train_datasets]),
                                        batch_size=total_batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=False)

            out_val_loader = DataLoader(dataset=ConcatDataset([dataset for dataset in val_datasets]),
                                         batch_size=total_batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False)
            
            test_loader = DataLoader(dataset=ConcatDataset([dataset for dataset in test_datasets]),
                                         batch_size=total_batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False)

            loaders.append((train_loader, in_val_loader, out_val_loader, test_loader))


    elif cfgs['test_dom'] == "IID":
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for fold in dom_list:
            dataset = datasets_dict[cfgs['dataset']](fold, cfgs)

            train_dataset, val_dataset, test_dataset = random_split(dataset, 
                                                                    [int(cfgs['train_split']*len(dataset)), 
                                                                     len(dataset)-int(cfgs['train_split']*len(dataset))-int(cfgs['test_split']*len(dataset)),
                                                                     int(cfgs['test_split']*len(dataset))],
                                                                    generator=generator)

            train_weights = make_weights_for_balanced_classes(train_dataset)

            train_datasets.append((train_dataset, train_weights))
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)

        total_batch_size = len(train_datasets)*cfgs['batch_size']

        train_loaders = [InfiniteDataLoader(dataset=dataset,
                                            weights=weights,
                                            batch_size=cfgs['batch_size'],
                                            num_workers=int(args.num_workers/len(train_dataset)))
                        for dataset, weights in train_datasets]
        train_loaders = zip(*train_loaders)

        in_val_loaders = [DataLoader(dataset=dataset,
                                     batch_size=total_batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=False)
                          for dataset, _ in train_datasets]
        out_val_loaders = [DataLoader(dataset=dataset,
                                     batch_size=total_batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=False)
                           for dataset in val_datasets]
        test_loaders = [DataLoader(dataset=dataset,
                                   batch_size=total_batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=False)
                        for dataset in test_datasets]

        loaders.append((train_loaders, in_val_loaders, out_val_loaders, test_loaders))        


    else:
        # Single test_dom case
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for fold in dom_list:
            dataset = datasets_dict[cfgs['dataset']](fold, cfgs)
            if fold[0] in cfgs['test_dom']:
                test_datasets.append(dataset)

            else:
                train_dataset, val_dataset = random_split(dataset, 
                                                            [int(cfgs['train_split']*len(dataset)), 
                                                             len(dataset)-int(cfgs['train_split'])*len(dataset)],
                                                            generator=generator)

                train_weights = make_weights_for_balanced_classes(train_dataset)

                train_datasets.append((train_dataset, train_weights))
                val_datasets.append(val_dataset)

        total_batch_size = len(train_datasets)*cfgs['batch_size']

        train_loaders = [InfiniteDataLoader(dataset=dataset,
                                            weights=weights,
                                            batch_size=cfgs['batch_size'],
                                            num_workers=int(args.num_workers/len(train_dataset)))
                        for dataset, weights in train_datasets]
        train_loaders = zip(*train_loaders)

        in_val_loaders = [DataLoader(dataset=dataset,
                                     batch_size=total_batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=False)
                          for dataset, _ in train_datasets]
        out_val_loaders = [DataLoader(dataset=dataset,
                                      batch_size=total_batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=False)
                           for dataset in val_datasets]
        test_loaders = [DataLoader(dataset=dataset,
                                   batch_size=total_batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=False)
                        for dataset in test_datasets]

        loaders.append((train_loaders, in_val_loaders, out_val_loaders, test_loaders))

    return loaders
       
































