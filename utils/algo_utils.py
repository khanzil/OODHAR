import torch
import torch.nn as nn
import numpy as np
from utils.networks_utils import Featurizer, Classifier, GRL

class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, cfg):
        super(Algorithm, self).__init__()
        self.cfg = cfg

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
    def __init__ (self, input_shape, cfg):
        super().__init__()
        self.featurizer = Featurizer(input_shape, cfg)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfg['dataset']['num_classes'],
            cfg["model"]["nonliner_classifier"]
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          lr=cfg['model']['learning_rate'],
                                          weight_decay=cfg['model']['weight_decay'])
        
        if cfg['model']['loss_type'] is 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfg['model']['loss_type']} is not implemented")
        
    def update(self, minibatch, unlabeled=None):
        all_x = torch.cat([x for x, y, d in minibatch])
        all_y = torch.cat([y for x, y, d in minibatch])
        loss = self.loss_type(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def validate(self, minibatch):
        
        all_x = torch.cat([x for x, y, d in minibatch])
        all_y = torch.cat([y for x, y, d in minibatch])
        with torch.no_grad:
            self.featurizer.eval()
            self.classifier.eval()
            loss = self.loss_type(self.predict(all_x), all_y)

        self.featurizer.train()
        self.classifier.train()

        return {'loss': loss.item()}

    def save_ckpt(self, epoch, checkpoint_path):
        state_dict = {
            'epoch': epoch,
            'network': self.network.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(),
            'np_random': np.random.get_state(),
        }
        torch.save(state_dict, checkpoint_path)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        epoch = state_dict['epoch']
        self.network.module.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return epoch


class DANN(Algorithm):
    def __init__ (self, input_shape, cfg):
        super().__init__()
        
        self.lambd = cfg['model']['lambda']
        self.featurizer = Featurizer(input_shape, cfg)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfg['dataset']['num_classes'],
            cfg["model"]["nonlinear_classifier"]
        )

        self.discriminator = nn.Sequential(
            GRL(cfg['model']['lambda']),
            Classifier(
            self.featurizer.n_outputs,
            cfg['dataset']['num_domains'],
            cfg["model"]["nonlinear_discriminator"]
            )
        )
        self.optimizer = torch.optim.Adam((list(self.featurizer.parameters())+list(self.classifier.parameters())), 
                                          lr=cfg['model']['learning_rate'],
                                          weight_decay=cfg['model']['weight_decay'])
        
        if cfg['model']['loss_type'] is 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfg['model']['loss_type']} is not implemented")
        
        if cfg['model']['loss_type_d'] is 'CrossEntropy':
            self.loss_type_d = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfg['model']['loss_type_d']} is not implemented")        
        
    def update(self, minibatch):
        all_x = torch.cat([x for x, y, d in minibatch])
        all_y = torch.cat([y for x, y, d in minibatch])
        all_d = torch.cat([d for x, y, d in minibatch])

        all_z = self.featurizer(all_x)

        loss_class = self.loss_type(self.classifier(all_z), all_y)
        loss_domain = self.loss_type_d(self.discriminator(all_z), all_d) 
        loss = loss_class + loss_domain * self.lambd

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    

        return {'loss': loss.item(), 
                'loss_class': loss_class.item(), 
                'loss_domain': loss_domain.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def validate(self, minibatch):
        all_x = torch.cat([x for x, y, d in minibatch])
        all_y = torch.cat([y for x, y, d in minibatch])
        all_d = torch.cat([d for x, y, d in minibatch])

        all_z = self.featurizer(all_x)

        with torch.no_grad():
            self.featurizer.eval()
            self.classifier.eval()
            self.discriminator.eval()

            loss_class = self.loss_type(self.classifier(all_z), all_y)
            loss_domain = self.loss_type_d(self.discriminator(all_z), all_d) 
            loss = loss_class + loss_domain * self.lambd

        self.featurizer.train()
        self.classifier.train()
        self.discriminator.train()

        return {'loss': loss.item(), 
                'loss_class': loss_class.item(), 
                'loss_domain': loss_domain.item()}
    def save_ckpt(self, epoch, checkpoint_path):
        state_dict = {
            'epoch': epoch,
            'featurizer': self.featurizer.module.state_dict(),
            'classifier': self.classifier.module.state_dict(),
            'discriminator': self.discriminator.modeul.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(),
            'np_random': np.random.get_state(),
        }
        torch.save(state_dict, checkpoint_path)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        epoch = state_dict['epoch']
        self.network.module.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return epoch



















