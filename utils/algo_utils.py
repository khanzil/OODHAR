import torch
import torch.nn as nn
import numpy as np
import os
from utils.networks_utils import Featurizer, Classifier, GRL

class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self):
        super().__init__()

    def update(self, minibatch, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatch from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError

    def save_ckpt(self):
        raise NotImplementedError

    def load_ckpt(self):
        raise NotImplementedError

class ERM(Algorithm):
    def __init__ (self, cfg):
        super().__init__()
        self.cfg = cfg
        self.featurizer = Featurizer(cfg)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfg['dataset']['num_classes'],
            cfg["model"]["nonlinear_classifier"]
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          lr=cfg['algo']['learning_rate'],
                                          weight_decay=cfg['algo']['weight_decay'])
        
        if cfg['algo']['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfg['algo']['loss_type']} is not implemented")
        
        self.loss_dict = {'loss_class': 0}

        self.loss_dict_val = {'loss_class': 0}
        
    def update(self, minibatch, unlabeled=None):
        all_x = minibatch.batch_feature
        all_y = minibatch.batch_label
        loss = self.loss_type(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_dict['loss_class'] += loss.item()

    def predict(self, x):
        return self.network(x)

    def validate(self, minibatch):  
        all_x = minibatch.batch_feature
        all_y = minibatch.batch_label
        self.featurizer.eval()
        self.classifier.eval()

        with torch.no_grad():
            pred = self.predict(all_x)
            loss_class = self.loss_type(pred, all_y)

            _, pred = pred.max(1) # same as np.argmax()
            num_corrects = torch.eq(pred, all_y).sum()

        self.featurizer.train()
        self.classifier.train()

        self.loss_dict_val['loss_class'] += loss_class.item()
        return num_corrects

    def save_ckpt(self, epoch, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'ckpt.pth.rar')
        state_dict = {
            'epoch': epoch,
            'network': self.network.modules.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(),
            'np_random': np.random.get_state(),
        }
        torch.save(state_dict, checkpoint_dir)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        epoch = state_dict['epoch']
        self.network.modules.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return epoch


class DANN(Algorithm):
    def __init__ (self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lambd = cfg['algo']['lambda']
        self.featurizer = Featurizer(cfg)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfg['dataset']['num_classes'],
            cfg["model"]["nonlinear_classifier"]
        )

        self.discriminator = nn.Sequential(
            GRL(),
            Classifier(
            self.featurizer.n_outputs,
            cfg['dataset']['num_domains'],
            cfg["algo"]["nonlinear_discriminator"]
            )
        )
        self.optimizer = torch.optim.Adam((list(self.featurizer.parameters())+list(self.classifier.parameters())+list(self.discriminator.parameters())), 
                                          lr=cfg['algo']['learning_rate'],
                                          weight_decay=cfg['algo']['weight_decay'])
        
        if cfg['algo']['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfg['algo']['loss_type']} is not implemented")
        
        if cfg['algo']['loss_type_d'] == 'CrossEntropy':
            self.loss_type_d = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfg['algo']['loss_type_d']} is not implemented")
        
        self.loss_dict = {'loss': 0,
                          'loss_class': 0,
                          'loss_domain': 0}
        
        self.loss_dict_val = {'loss_class': 0}
        
    def update(self, minibatch):
        all_x = minibatch.batch_feature
        all_y = minibatch.batch_label
        all_d = minibatch.batch_domain

        all_z = self.featurizer(all_x)

        pred = self.classifier(all_z)
        pred_d = self.discriminator(all_z)

        loss_class = self.loss_type(pred, all_y)
        loss_domain = self.loss_type_d(pred_d, all_d) 
        loss = loss_class + loss_domain * self.lambd

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    

        self.loss_dict['loss'] += loss.item()
        self.loss_dict['loss_class'] += loss_class.item()
        self.loss_dict['loss_domain'] += loss_domain.item()

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def validate(self, minibatch):
        all_x = minibatch.batch_feature
        all_y = minibatch.batch_label

        self.featurizer.eval()
        self.classifier.eval()

        with torch.no_grad():
            pred = self.predict(all_x)
            loss_class = self.loss_type(pred, all_y)

            _, pred = pred.max(1) # same as np.argmax()
            num_corrects = torch.eq(pred, all_y).sum()

        self.featurizer.train()
        self.classifier.train()

        self.loss_dict_val['loss_class'] += loss_class.item()
        return num_corrects
    
    def save_ckpt(self, epoch, results_dir):
        checkpoint_path = os.path.join(results_dir, 'ckpts' ,f'Epoch_{epoch}_ckpt.pth.rar')
        state_dict = {
            'epoch': epoch,
            'featurizer': self.featurizer.modules.state_dict(),
            'classifier': self.classifier.modules.state_dict(),
            'discriminator': self.discriminator.modules.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(),
            'np_random': np.random.get_state(),
        }
        torch.save(state_dict, checkpoint_path)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        epoch = state_dict['epoch']
        self.featurizer.modules.load_state_dict(state_dict['featurizer'])
        self.classifier.modules.load_state_dict(state_dict['classifier'])
        self.discriminator.modules.load_state_dict(state_dict['discriminator'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return epoch



















