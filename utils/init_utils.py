import os
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from utils.algo_utils import *
from utils.dataset_utils import *

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



def get_algo(cfgs, args):
    return algos_dict[cfgs['algorithm']](cfgs, args)

def get_dataloader(cfgs, args):
    dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset']) # directory contains all data folders
    dom_list = [fold for fold in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,fold))]
    loaders = []
    dataset_num_workers = -1

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
                if fold[0] in test_dom:
                    test_datasets.append(dataset)

                else:
                    idx = np.arange(len(dataset))
                    np.random.shuffle(idx)
                    train_dataset = Subset(dataset, idx[:int(cfgs['train_split']*len(dataset))+1])
                    val_dataset = Subset(dataset, idx[int(cfgs['train_split']*len(dataset))+1:])

                    train_weights = make_weights_for_balanced_classes(train_dataset)

                    train_datasets.append((train_dataset, train_weights))
                    val_datasets.append(val_dataset)

            total_batch_size = len(train_datasets)*cfgs['batch_size']

            train_loaders = [InfiniteDataLoader(dataset=dataset,
                                                weights=weights,
                                                batch_size=cfgs['batch_size'],
                                                num_workers=dataset_num_workers)
                            for dataset, weights in train_datasets]
            train_loaders = zip(*train_loaders)


            in_val_loaders = DataLoader(dataset=ConcatDataset([dataset for dataset, _ in train_datasets]),
                                        batch_size=total_batch_size,
                                        num_workers=args.num_workers)
            out_val_loaders = DataLoader(dataset=ConcatDataset(val_datasets),
                                    batch_size=total_batch_size,
                                    num_workers=args.num_workers)
            test_loaders = DataLoader(dataset=ConcatDataset(test_datasets),
                                    batch_size=total_batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False)

            loaders.append((train_loaders, in_val_loaders, out_val_loaders, test_loaders))



    elif cfgs['test_dom'] == "IID":
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for fold in dom_list:
            dataset = datasets_dict[cfgs['dataset']](fold, cfgs)

            idx = np.arange(len(dataset))
            np.random.shuffle(idx)
            train_dataset = Subset(dataset, idx[:int(cfgs['train_split']*len(dataset))])
            val_dataset = Subset(dataset, idx[int(cfgs['train_split']*len(dataset)):-int(cfgs['test_split']*len(dataset))])
            test_dataset = Subset(dataset, idx[-int(cfgs['test_split']*len(dataset)):])

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

        in_val_loaders = DataLoader(dataset=ConcatDataset([dataset for dataset, _ in train_datasets]),
                                    batch_size=total_batch_size,
                                    num_workers=args.num_workers)
        out_val_loaders = DataLoader(dataset=ConcatDataset(val_datasets),
                                 batch_size=total_batch_size,
                                 num_workers=args.num_workers)

        test_loaders = DataLoader(dataset=ConcatDataset(test_datasets),
                                  batch_size=total_batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False)

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
                idx = np.arange(len(dataset))
                np.random.shuffle(idx)
                train_dataset = Subset(dataset, idx[:int(cfgs['train_split']*len(dataset))])
                val_dataset = Subset(dataset, idx[int(cfgs['train_split']*len(dataset)):])

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

        in_val_loaders = DataLoader(dataset=ConcatDataset([dataset for dataset, _ in train_datasets]),
                                    batch_size=total_batch_size,
                                    num_workers=args.num_workers)
        out_val_loaders = DataLoader(dataset=ConcatDataset(val_datasets),
                                 batch_size=total_batch_size,
                                 num_workers=args.num_workers)

        test_loaders = DataLoader(dataset=ConcatDataset(test_datasets),
                                  batch_size=total_batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False)

        loaders.append((train_loaders, in_val_loaders, out_val_loaders, test_loaders))

    return loaders
       
































