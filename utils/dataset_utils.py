import torch
from torch.utils.data import Dataset
import h5py

class Glasgow(Dataset):
    def __init__(self, folds, cfg):
        super().__init__()
        dataset_dir = cfg['dataset']['rootdir'] + '/' + cfg['dataset']['dataset']
        self.folds = folds
        self.features = []
        self.label = []

        for i, fold in enumerate(self.folds):
            feature_h5_dir = dataset_dir + '/' + fold + '/extracted_feature.h5'
            label_h5_dir = dataset_dir + '/' + fold + '/extracted_label.h5'
            with h5py.File(feature_h5_dir, 'r') as hf:
                feature = hf[cfg['dataset']['feature_type']][()] # this has size (samples, range, time) or (samples, range, vel)
            with h5py.File(label_h5_dir, 'r') as hf:
                label = hf['label'][()]
                # person = hf['person'][()]

            domain = torch.ones(len(label)) * i
            self.features.append(feature)
            self.label.append(torch.stack((label, domain)))
            
        self.features = torch.cat(self.features)
        self.label = torch.cat(self.label)

    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, index):
        feature = self.features[index,...]
        label = self.label[index,0,:]
        domain = self.label[index,1,:]
        # person = self.label[index,2,:]

        sample = {
            'feature'   : feature,
            'label'     : label,
            'domain'    : domain,
            # 'person'    : person,
        }
        return sample

class GlasgowCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        batch_feature = []
        batch_label = []
        batch_domain = []
        # batch_person = []

        for i in range(len(batch)):
            batch_feature.append(batch[i]['feature'])
            batch_label.append(batch[i]['label'])
            batch_domain.append(batch[i]['domain'])
            # batch_person.append(batch[i]['person'])

        batch_feature = torch.cat(batch_feature)
        batch_label = torch.cat(batch_label)
        batch_domain = torch.cat(batch_domain)
        # batch_person = torch.cat(batch_person)
        
        return batch_feature, batch_label, batch_domain#, batch_person



        

























