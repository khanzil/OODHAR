import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms.functional import rotate
import h5py
import os
import numpy as np
from collections import Counter

class Glasgow():
    # This class acts as an multi-domain dataset holder

    # Subclass can override these
    num_workers = 2    # Because we are using multiple loaders in this code, num_workers should be pre-calculated and depends on dataset
    num_val_workers = 4
    batch_size = 16
    num_steps = 1200
    val_freq = 20
    ckpt_freq = 100
    domain_key = 'folder' # key to extract domain from .h5 file
    num_classes = 6
    num_train_domains = 6
    num_domains = 7
    feature_type = 'v_heatmap'
    domain_key = 'folder'

    def __init__(self, cfgs):
        self.cfgs = cfgs
        dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset'])
        folds = [fold for fold in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir,fold))]
        self.datasets = [GlasgowFolderDataset(dataset_dir,fold,self.feature_type,self.domain_key) for fold in folds]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]

    def set_cfgs(self):
        '''
            Set the parameters into cfgs so that it got carried on to other functions 
        '''
        self.cfgs['batch_size'] = self.batch_size
        self.cfgs['num_steps'] = self.num_steps
        self.cfgs['val_freq'] = self.val_freq
        self.cfgs['ckpt_freq'] = self.ckpt_freq
        self.cfgs['domain_key'] = self.domain_key
        self.cfgs['num_classes'] = self.num_classes
        self.cfgs['num_train_domains'] = self.num_train_domains
        self.cfgs['num_domains'] = self.num_domains
        self.cfgs['feature_type'] = self.feature_type 
        self.cfgs['domain_key'] = self.domain_key 

class GlasgowAge(Glasgow):
    num_workers = 4
    batch_size = 32
    num_steps = 2000
    val_freq = 40
    ckpt_freq = 200
    domain_key = 'age_cat' # key to extract domain from .h5 file
    num_classes = 6
    num_train_domains = 2
    num_domains = 3
    domain_key = 'age_cat'

class SanityCheck(Glasgow):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.num_steps = 200


class GlasgowFolderDataset(Dataset):
    def __init__(self, dataset_dir, fold, feature_type, domain_key):
        super().__init__()
        self.feature_type = feature_type
        self.domain_key = domain_key
        self.fold = fold
        self.fold_dir = os.path.join(dataset_dir,fold)
        self.paths = np.array([path for path in os.listdir(self.fold_dir) if os.path.isfile(os.path.join(self.fold_dir,path)) and ".h5" in path])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        return self.file_loader_h5py(os.path.join(self.fold_dir, path))

    def get_sample_weights(self, idx):
        all_y = torch.tensor([int(path[path.find('A')+1:path.find('R')])-1 for path in self.paths[idx]])
        counts = torch.bincount(all_y)
        return 1.0 / counts[all_y].float()

    def file_loader_h5py(self, path):
        with h5py.File(path, 'r') as hf:
            x = hf[self.feature_type][()] # this has size (samples, range, time) or (samples, range, vel)
            y = hf['label'][()] - 1
            d = hf[self.domain_key][()]
        
        x = torch.abs(torch.from_numpy(x)).float()
        if len(x.shape) == 2:
            x = x.unsqueeze(0) # when using conv2d, it is necessary for input sample to have n_dims = 3

        y = torch.tensor(y).long()
        d = torch.tensor(d).long()

        return x, y, d    


class MNISTBased(Dataset):
    num_workers = 4    # Because we are using multiple loaders in this code, num_workers should be calculated and depends on dataset
    num_val_workers = 4
    batch_size = 32
    num_steps = 5000
    val_freq = 200
    ckpt_freq = 500
    num_classes = 10
    domains = ['+90%', '+80%', '-90%']
    num_domains = 3
    num_train_domains = 2


    def __init__(self, cfgs):
        self.cfgs = cfgs
        dataset_dir = os.path.join(cfgs['rootdir'], cfgs['dataset'])
        original_dataset_tr = MNIST(dataset_dir, train=True, download=True)
        original_dataset_te = MNIST(dataset_dir, train=False, download=True)
        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        
        self.datasets = []

        for i in range(len(self.domains)):
            images = original_images[i::len(self.domains)]
            labels = original_labels[i::len(self.domains)]
            self.datasets.append(self.dataset_transform(images, labels, self.domains[i]))

    def dataset_transform(self, images, labels):
        pass

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
    def __init__(self, dataset, weights, batch_size, num_workers, generator):
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
            pin_memory=True,
            generator=generator
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

# Modified from DomainBed
def make_weights_for_balanced_classes(dataset):
    # counts = Counter()
    # classes = []
    # for _, y in dataset:
    #     y = int(y)
    #     counts[y] += 1
    #     classes.append(y)

    # n_classes = len(counts)

    # weight_per_class = {}
    # for y in counts:
    #     weight_per_class[y] = 1 / (counts[y] * n_classes)

    # weights = torch.zeros(len(dataset))
    # for i, y in enumerate(classes):
    #     weights[i] = weight_per_class[int(y)]

    if isinstance(dataset, torch.utils.data.Subset):
        try:
            idx = torch.tensor(dataset.indices)
            return dataset.dataset.get_sample_weights(idx)
        except:
            pass
    if isinstance(dataset, torch.utils.data.Dataset):
        try:
            return dataset.get_sample_weights(torch.arange(len(dataset)))
        except:
            pass

    # all_y = []
    # for _, y, _ in dataset:
    #     all_y.append(y)
    # counts = torch.bincount(torch.Tensor(all_y))
    # weights = 1.0 / counts[all_y].float()

    # return weights


























