import torch
from torch.utils.data import Dataset
import h5py
import os
from collections import Counter

class Glasgow(Dataset):
    def __init__(self, fold, cfgs):
        super().__init__()
        dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset'])
        # Because we are using multiple loaders in this code, num_workers should be calculated and dependent on dataset
        # 
        self.num_workers = 1
        self.cfgs = cfgs
        self.fold = fold
        self.fold_dir = os.path.join(dataset_dir,fold)

        self.paths = [path for path in os.listdir(self.fold_dir) if os.path.isfile(os.path.join(self.fold_dir,path)) and ".h5" in path]


    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        sample, target, domain = self.file_loader_h5py(os.path.join(self.fold_dir, path))

        return sample, target, domain

    def file_loader_h5py(self, path):
        with h5py.File(path, 'r') as hf:
            x = hf[self.cfgs['feature_type']][()] # this has size (samples, range, time) or (samples, range, vel)
            y = hf['label'][()] - 1
            d = hf[self.cfgs['domain']][()]
        
        x = torch.abs(torch.from_numpy(x)).float()
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        y = torch.tensor(y).long()
        d = torch.tensor(d).long()

        return x, y, d
        


# These were taken from DomainBed
class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=True
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y, _ in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


























