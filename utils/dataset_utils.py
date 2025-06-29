import torch
from torch.utils.data import Dataset
import h5py
import os

class Glasgow(Dataset):
    def __init__(self, folds, cfg):
        super().__init__()
        dataset_dir = os.path.join(cfg['dataset']['rootdir'], cfg['dataset']['dataset'])
        self.folds = folds
        self.all_x = []
        self.all_y = []
        self.all_d = []

        for fold in self.folds:
            for file in os.listdir(os.path.join(dataset_dir,fold)):
                feature_h5_dir = os.path.join(dataset_dir, fold, file)
                with h5py.File(feature_h5_dir, 'r') as hf:
                    feature = hf[cfg['dataset']['feature_type']][()] # this has size (samples, range, time) or (samples, range, vel)
                    y = hf['label'][()]
                    d = hf[cfg['dataset']['domain']][()]

                self.all_x.append(torch.abs(torch.from_numpy(feature)))
                self.all_y.append(torch.tensor(y)-1)
                self.all_d.append(torch.tensor(d))

        self.all_x = torch.stack(self.all_x).float()
        if len(self.all_x.shape) == 3:
            self.all_x = self.all_x.unsqueeze(1)
        self.all_y = torch.stack(self.all_y).long()
        self.all_d = torch.stack(self.all_d).long()

    def __len__(self):
        return self.all_y.shape[0]
    
    def __getitem__(self, index):
        feature = self.all_x[index,...]
        label = self.all_y[index]
        domain = self.all_d[index]

        sample = {
            'feature'   : feature,
            'label'     : label,
            'domain'    : domain,
        }
        return sample

class GlasgowCollateCustom:
    def __init__(self, batch):
        self.batch_feature = []
        self.batch_label = []
        self.batch_domain = []
        # batch_person = []

        for i in range(len(batch)):
            self.batch_feature.append(batch[i]['feature'])
            self.batch_label.append(batch[i]['label'])
            self.batch_domain.append(batch[i]['domain'])
            # batch_person.append(batch[i]['person'])

        self.batch_feature = torch.stack(self.batch_feature)
        self.batch_label = torch.tensor(self.batch_label)
        self.batch_domain = torch.tensor(self.batch_domain)
        # batch_person = torch.cat(batch_person)
        
    def pin_memory(self):
        self.batch_feature = self.batch_feature.pin_memory()
        self.batch_label = self.batch_label.pin_memory()
        self.batch_domain = self.batch_domain.pin_memory()

def GlasgowCollate(batch):
    return GlasgowCollateCustom(batch)

class GlasgowCollateTest:
    pass

        

























