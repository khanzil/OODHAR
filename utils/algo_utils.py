import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from utils.networks_utils import Featurizer, Classifier, GRL
import json

class Algorithm():
    '''
        This acts as a Trainer
    '''
    def __init__(self):
        pass

    def train_step(self, train_loader, epoch, unlabeled=None):
        '''
            Perform 1 epoch update, return a dictionary of mean(loss)
        '''
        raise NotImplementedError

    def train(self, num_epochs, train_loader, val_loader, test_loader, ckpt_crit='acc', ckpt_freq=10, results_dir=None, cur_epoch=0):
        '''
            Trainer function that performs training over num_epochs epochs.
        '''
        loss_list = []
        best_score = 0.0

        iterator = tqdm(range(cur_epoch, num_epochs), total=num_epochs-cur_epoch, unit='epoch', position=0, leave=True)
        for epoch in iterator:

            '''
                Perform training
            '''
            loss_list.append(self.train_step(train_loader, epoch))

            '''
                Calculate metrics on validation set and train.
            '''
            _, train_acc = self.validate_step(train_loader)
            _, val_acc = self.validate_step(val_loader)
            _, test_acc = self.validate_step(test_loader)

            loss_list[-1].update({'train_acc': train_acc,
                                  'val_acc': val_acc,
                                  'test_acc': test_acc,
                                  'epoch': float(epoch+1)})
            
            '''
                Print and save validation results after every epoch
            '''
            for key in loss_list[-1].keys():
                tqdm.write(f"{key}".ljust(15), end = "")
            tqdm.write("")

            for key in loss_list[-1].keys():
                tqdm.write(f"{loss_list[-1][key]:.10f}".ljust(15), end="")
            tqdm.write("")

            '''
                Save the checkpoints
            '''
            if (epoch+1) % ckpt_freq == 0: 
                self.save_ckpt(epoch, results_dir)


            if val_acc > best_score:
                best_score = val_acc
                self.save_ckpt(epoch, results_dir, is_best=True)
                
        output_file = open(os.path.join(results_dir, 'loss_list'), 'a', encoding='utf-8')
        for dic in loss_list:
            json.dump(dic, output_file)
            output_file.write("\n")
        
        return loss_list

    def predict(self, x):
        '''
            Perform 1 minibatch predict and calculate loss, add loss to self.loss_dict['train']
        '''
        raise NotImplementedError

    def validate_step(self, loader):
        '''
            loader: dataloader for validation
            trainval: 'train' or 'val'

            Perform validation over the entire loader, the results are stored in self.loss_dict[trainval].
            Note that in this function, acc is calculated by getting the number of correct predictions then divided by loader_len.

            In case validating on training set, train_loss will not be recalculated.
        '''
        raise NotImplementedError

    def save_ckpt(self):
        raise NotImplementedError

    def load_ckpt(self):
        raise NotImplementedError   

class ERM(Algorithm):
    def __init__ (self, cfg, args):
        self.cuda = args.cuda
        self.featurizer = Featurizer(cfg)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfg['num_classes'],
            cfg['nonlinear_classifier']
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          lr=cfg['learning_rate'],
                                          weight_decay=cfg['weight_decay'])
        
        if cfg['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfg['loss_type']} is not implemented")
       
        if self.cuda:
            self.featurizer.cuda()
            self.classifier.cuda()

    def train_step(self, train_loader, epoch, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()
        loader_len = 0.0
        total_loss_class = 0.0

        for batch_idx, minibatch in enumerate(train_loader):
            all_x = minibatch.batch_feature
            all_y = minibatch.batch_label
            if self.cuda:
                all_x = all_x.cuda()
                all_y = all_y.cuda()

            loss_class = self.loss_type(self.predict(all_x), all_y)

            self.optimizer.zero_grad()
            loss_class.backward()
            self.optimizer.step()

            total_loss_class += loss_class.item()
            loader_len += all_x.shape[0]

        total_loss_class /= loader_len

        return {'loss_class' : total_loss_class}

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        self.featurizer.eval()
        self.classifier.eval()
        acc = 0.0
        loader_len = 0.0

        pred_list = []

        for batch_idx, minibatch in enumerate(loader):
            all_x = minibatch.batch_feature
            all_y = minibatch.batch_label

            if self.cuda:
                all_x = all_x.cuda()
                all_y = all_y.cuda()

            with torch.no_grad():
                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                num_corrects = torch.eq(pred, all_y).sum()
                pred_list.extend(zip(pred.cpu().numpy(),all_y.cpu().numpy()))

                acc += num_corrects.cpu().numpy()      
                loader_len += all_x.shape[0]

        self.featurizer.train()
        self.classifier.train()

        return pred_list, acc/loader_len

    def save_ckpt(self, epoch, results_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(results_dir, 'ckpts' ,f'Best_ckpt.pth.rar')
        else:
            checkpoint_path = os.path.join(results_dir, 'ckpts' ,f'Epoch_{epoch}_ckpt.pth.rar')

        state_dict = {
            'epoch': epoch,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'np_random': np.random.get_state(),
        }
        if torch.cuda.is_available():
            state_dict.update({'cuda_rng': torch.cuda.get_rng_state()})
        torch.save(state_dict, checkpoint_path)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=False)
        epoch = state_dict['epoch']
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return epoch

class DANN(Algorithm):
    def __init__ (self, cfg, args):
        self.cuda = args.cuda
        self.lambd = cfg['lambd']
        self.d_steps_per_g_step = cfg['d_steps_per_g_step']
        self.featurizer = Featurizer(cfg)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfg['num_classes'],
            cfg['nonlinear_classifier']
        )

        self.discriminator = nn.Sequential(
            GRL(),
            Classifier(
            self.featurizer.n_outputs,
            cfg['num_domains'],
            cfg['nonlinear_discriminator']
            )
        )
        self.optimizer = torch.optim.Adam((list(self.featurizer.parameters())+list(self.classifier.parameters())+list(self.discriminator.parameters())), 
                                          lr=cfg['learning_rate'],
                                          weight_decay=cfg['weight_decay'])
        
        if cfg['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfg['loss_type']} is not implemented")
        
        if cfg['loss_type_d'] == 'CrossEntropy':
            self.loss_type_d = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfg['loss_type_d']} is not implemented")
        
        if self.cuda:
            self.featurizer.cuda()
            self.classifier.cuda()
            self.discriminator.cuda()

    def train_step(self, train_loader, epoch, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()
        self.discriminator.train()

        total_loss = 0.0
        total_loss_class = 0.0
        total_loss_domain = 0.0
        loader_len = 0.0
        
        for batch_idx, minibatch in enumerate(train_loader):
            all_x = minibatch.batch_feature
            all_y = minibatch.batch_label
            all_d = minibatch.batch_domain
            if self.cuda:
                all_x = all_x.cuda()
                all_y = all_y.cuda()
                all_d = all_d.cuda()

            all_z = self.featurizer(all_x)

            pred = self.classifier(all_z)
            pred_d = self.discriminator(all_z)

            loss_class = self.loss_type(pred, all_y)
            loss_domain = self.loss_type_d(pred_d, all_d)
            if epoch % self.d_steps_per_g_step == 0:
                loss = loss_class + loss_domain * self.lambd
            else:
                loss = loss_class

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_loss_class += loss_class.item()
            total_loss_domain += loss_domain.item()
            loader_len += all_x.shape[0]

        total_loss /= loader_len
        total_loss_class /= loader_len
        total_loss_domain /= loader_len

        return {'loss': total_loss, 
                'loss_class': total_loss_class,
                'loss_domain': total_loss_domain}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def validate_step(self, loader):
        self.featurizer.eval()
        self.classifier.eval()
        pred_list = []
        acc = 0.0
        loader_len = 0.0

        for batch_idx, minibatch in enumerate(loader):
            all_x = minibatch.batch_feature
            all_y = minibatch.batch_label
            all_d = minibatch.batch_domain

            if self.cuda:
                all_x = all_x.cuda()
                all_y = all_y.cuda()
                all_d = all_d.cuda()

            with torch.no_grad():
                all_z = self.featurizer(all_x)
                pred = self.classifier(all_z)
                _, pred = pred.max(1) # same as np.argmax()
                num_corrects = torch.eq(pred, all_y).sum()
                pred_list.extend(zip(pred.cpu().numpy(),all_y.cpu().numpy()))

            acc += num_corrects.cpu().numpy()
            loader_len += all_x.shape[0]

        self.featurizer.train()
        self.classifier.train()

        return pred_list, acc/loader_len

    def save_ckpt(self, epoch, results_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(results_dir, 'ckpts' ,f'Best_ckpt.pth.rar')
        else:
            checkpoint_path = os.path.join(results_dir, 'ckpts' ,f'Epoch_{epoch}_ckpt.pth.rar')

        state_dict = {
            'epoch': epoch,
            'featurizer': self.featurizer.state_dict(),
            'classifier': self.classifier.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'np_random': np.random.get_state(),
        }
        if torch.cuda.is_available():
            state_dict.update({'cuda_rng': torch.cuda.get_rng_state()})
        torch.save(state_dict, checkpoint_path)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=False)
        epoch = state_dict['epoch']
        self.featurizer.load_state_dict(state_dict['featurizer'])
        self.classifier.load_state_dict(state_dict['classifier'])
        self.discriminator.load_state_dict(state_dict['discriminator'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return epoch



















