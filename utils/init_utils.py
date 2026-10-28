import os
import numpy as np
import hashlib
from torch.utils.data import DataLoader, random_split, ConcatDataset

from utils.algo_utils import *
from utils.dataset_utils import *
from utils.sweep_utils import get_random_search_configs

algos_dict = {
    'ERM'       : ERM,
    'DANN'      : DANN,
    'IRM'       : IRM,
    'Fish'      : Fish,
    'VRex'      : VRex,
    'CFSM'      : CFSM,
    'GroupDRO'  : GroupDRO,
    'Fishr'     : Fishr,
    'SAM'       : SAM,
    'MMD'       : MMD,
    'CORAL'     : CORAL,
    'Proposed1' : Proposed1,
    'Proposed2' : Proposed2,
}

datasets_dict = {
    'Glasgow'       : Glasgow,
    'GlasgowAge'    : GlasgowAge,
    'SanityCheck'   : SanityCheck,
}

def get_random_seed(*args):
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**32)

def get_algo(cfgs, args):
    return algos_dict[cfgs['algorithm']](cfgs, args)

def get_domain_datasets(cfgs, args):
    dom_datasets = datasets_dict[cfgs['dataset']](cfgs)
    dom_datasets.set_cfgs()
    return dom_datasets

def get_dataloaders(cfgs, args, dom_datasets):
    loaders = []
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    generator_loader = torch.Generator()
    generator_loader.manual_seed(args.search)

    # Warning: make_weights_for_balanced_classes() iterates through all train data to get weights, pre-calculate this to save time in initialization phase 
    if cfgs['test_dom'] == 'None':
        # This code is using train-valication domain split and sweep through all test domain combination
        for i_test_dom, test_dom in enumerate(dom_datasets):
            in_splits = [random_split(dataset, [cfgs['train_split'], 1-cfgs['train_split']],generator=generator)
                            for i_dataset, dataset in enumerate(dom_datasets) if i_dataset != i_test_dom]
            
            train_datasets = [(train,make_weights_for_balanced_classes(train)) for train,_ in in_splits]
            val_datasets = [val for _,val in in_splits]
            test_datasets = test_dom


            val_batch_size = len(train_datasets)*dom_datasets.batch_size


            train_loader = [InfiniteDataLoader(dataset=dataset,
                                                weights=weights,
                                                batch_size=dom_datasets.batch_size,
                                                num_workers=dom_datasets.num_workers,
                                                generator=generator_loader)
                             for dataset, weights in train_datasets]
            train_loader = zip(*train_loader)

            in_val_loader = DataLoader(dataset=ConcatDataset([dataset for dataset, _ in train_datasets]),
                                       batch_size=val_batch_size,
                                       num_workers=dom_datasets.num_val_workers,
                                       shuffle=False,
                                       pin_memory=True,
                                       persistent_workers=True)

            out_val_loader = DataLoader(dataset=ConcatDataset([dataset for dataset in val_datasets]),
                                        batch_size=val_batch_size,
                                        num_workers=dom_datasets.num_val_workers,
                                        shuffle=False,
                                        pin_memory=True,
                                        persistent_workers=True)
            
            test_loader = DataLoader(dataset=test_datasets,
                                     batch_size=val_batch_size,
                                     num_workers=dom_datasets.num_val_workers,
                                     shuffle=False,
                                     pin_memory=True,
                                     persistent_workers=True)

            loaders.append((train_loader, in_val_loader, out_val_loader, test_loader))


    elif cfgs['test_dom'] == "IID":
        in_splits = [random_split(dataset, [int(cfgs['train_split']*len(dataset)), 
                                         len(dataset)-int(cfgs['train_split']*len(dataset))-int(cfgs['test_split']*len(dataset)),
                                         int(cfgs['test_split']*len(dataset))], generator=generator)
                    for i_dataset, dataset in enumerate(dom_datasets)]
        
        train_datasets = [(train,make_weights_for_balanced_classes(train)) for train,_,_ in in_splits]
        val_datasets = [val for _,val,_ in in_splits]
        test_datasets = [test for _,_,test in in_splits]


        val_batch_size = len(train_datasets)*dom_datasets.batch_size

            
        train_loader = [InfiniteDataLoader(dataset=dataset,
                                            weights=weights,
                                            batch_size=dom_datasets.batch_size,
                                            num_workers=dom_datasets.num_workers,
                                            generator=generator_loader)
                            for dataset, weights in train_datasets]
        train_loader = zip(*train_loader)

        in_val_loader = DataLoader(dataset=ConcatDataset([dataset for dataset, _ in train_datasets]),
                                    batch_size=val_batch_size,
                                    num_workers=dom_datasets.num_val_workers,
                                    shuffle=False,
                                    pin_memory=True,
                                    persistent_workers=True)

        out_val_loader = DataLoader(dataset=ConcatDataset([dataset for dataset in val_datasets]),
                                    batch_size=val_batch_size,
                                    num_workers=dom_datasets.num_val_workers,
                                    shuffle=False,
                                    pin_memory=True,
                                    persistent_workers=True)
        
        test_loader = DataLoader(dataset=ConcatDataset([dataset for dataset in test_datasets]),
                                    batch_size=val_batch_size,
                                    num_workers=dom_datasets.num_val_workers,
                                    shuffle=False,
                                    pin_memory=True,
                                    persistent_workers=True)


        loaders.append((train_loader, in_val_loader, out_val_loader, test_loader))        


    elif cfgs['test_dom'] in range(len(dom_datasets)):
        # Single test_dom case
        in_splits = [random_split(dataset, [cfgs['train_split'], 1-cfgs['train_split']],generator=generator)
                                    for i_dataset, dataset in enumerate(dom_datasets) if i_dataset != cfgs['test_dom']]
                    
        train_datasets = [(train,make_weights_for_balanced_classes(train)) for train,_ in in_splits]
        val_datasets = [val for _,val in in_splits]
        test_datasets = dom_datasets[cfgs['test_dom']]


        val_batch_size = len(train_datasets)*dom_datasets.batch_size

        
        train_loader = [InfiniteDataLoader(dataset=dataset,
                                            weights=weights,
                                            batch_size=dom_datasets.batch_size,
                                            num_workers=dom_datasets.num_workers,
                                            generator=generator_loader)
                            for dataset, weights in train_datasets]
        train_loader = zip(*train_loader)

        in_val_loader = DataLoader(dataset=ConcatDataset([dataset for dataset, _ in train_datasets]),
                                    batch_size=val_batch_size,
                                    num_workers=dom_datasets.num_val_workers,
                                    shuffle=False,
                                    pin_memory=True,
                                    persistent_workers=True)

        out_val_loader = DataLoader(dataset=ConcatDataset([dataset for dataset in val_datasets]),
                                    batch_size=val_batch_size,
                                    num_workers=dom_datasets.num_val_workers,
                                    shuffle=False,
                                    pin_memory=True,
                                    persistent_workers=True)
        
        test_loader = DataLoader(dataset=test_datasets,
                                    batch_size=val_batch_size,
                                    num_workers=dom_datasets.num_val_workers,
                                    shuffle=False,
                                    pin_memory=True,
                                    persistent_workers=True)

        loaders.append((train_loader, in_val_loader, out_val_loader, test_loader))

    else:
        raise ValueError(f"Test domain {cfgs['test_dom']} not available!")

    return loaders
       
































