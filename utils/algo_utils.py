import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
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

    def update(self, minibatch, step, unlabeled=None):
        '''
            Perform 1 batch update, return a dictionary of mean(loss)
        '''
        raise NotImplementedError

    def train(self, num_steps, train_loader, in_val_loader, out_val_loader, test_loader, val_freq, ckpt_freq, results_dir=None, cur_step=0):
        '''
            Trainer function that performs training over num_steps steps.
        '''
        loss_list = []

        total_step = num_steps-cur_step
        iterator = tqdm(range(cur_step, num_steps), total=total_step, unit='step', position=0, leave=True)
        for step in iterator:
            minibatchess = next(train_loader)

            '''
                Perform training
            '''
            loss_list.append(self.update(minibatchess, step))

            '''
                Calculate metrics on validation set and train.
            '''
            if step % val_freq == 0 or step==total_step-1:
                _, train_acc = self.validate_step(in_val_loader)
                _, val_acc = self.validate_step(out_val_loader)
                _, test_acc = self.validate_step(test_loader)
                mem_gb = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
                
                loss_list[-1].update({'train_acc': train_acc,
                                    'val_acc': val_acc,
                                    'test_acc': test_acc,
                                    'step': float(step),
                                    'mem_gb': mem_gb})
                
                '''
                    Print and save validation results at every val step
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
            if step+1 % ckpt_freq == 0: 
                self.save_ckpt(step, results_dir)
            
        output_file = open(os.path.join(results_dir, 'loss_list.json'), 'a', encoding='utf-8')
        for dic in loss_list:
            json.dump(dic, output_file)
            output_file.write("\n")
        
        return loss_list

    def predict(self, x):
        '''
            Perform 1 minibatch prediction
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
    def __init__ (self, cfgs, args):
        self.cuda = args.cuda
        self.featurizer = Featurizer(cfgs)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_classes'],
            cfgs['nonlinear_classifier']
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          lr=cfgs['learning_rate'],
                                          weight_decay=cfgs['weight_decay'])
        
        if cfgs['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")
       
        if self.cuda:
            self.featurizer.cuda()
            self.classifier.cuda()

    def update(self, minibatches, step, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()

        all_x = torch.cat([x for x,_,_ in minibatches])
        all_y = torch.cat([y for _,y,_ in minibatches])
        len_minibatch = len(all_y)

        if self.cuda:
            all_x = all_x.cuda()
            all_y = all_y.cuda()

        loss_class = self.loss_type(self.predict(all_x), all_y)
        
        self.optimizer.zero_grad()
        loss_class.backward()
        self.optimizer.step()

        return {'loss_class' : loss_class.item()/len_minibatch}

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        self.featurizer.eval()
        self.classifier.eval()
        acc = 0.0
        loader_len = 0.0

        pred_list = []

        for batch_idx, (all_x, all_y, all_d) in enumerate(loader):
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
    def __init__ (self, cfgs, args):
        self.cuda = args.cuda
        self.lambd = cfgs['DANN']['lambd']
        self.d_steps_per_g_step = cfgs['DANN']['d_steps_per_g_step']
        self.featurizer = Featurizer(cfgs)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_classes'],
            cfgs['nonlinear_classifier']
        )

        self.discriminator = nn.Sequential(
            GRL(),
            Classifier(
            self.featurizer.n_outputs,
            cfgs['num_domains'],
            cfgs['DANN']['nonlinear_discriminator']
            )
        )
        self.optimizer = torch.optim.Adam((list(self.featurizer.parameters())+list(self.classifier.parameters())+list(self.discriminator.parameters())), 
                                          lr=cfgs['learning_rate'],
                                          weight_decay=cfgs['weight_decay'])
        
        if cfgs['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")
        
        if cfgs['DANN']['loss_type_d'] == 'CrossEntropy':
            self.loss_type_d = nn.CrossEntropyLoss() 
        else:
            raise NotImplementedError(f"{cfgs['DANN']['loss_type_d']} is not implemented")
        
        if self.cuda:
            self.featurizer.cuda()
            self.classifier.cuda()
            self.discriminator.cuda()

    def update(self, minibatches, step, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()
        self.discriminator.train()

        all_x = torch.cat([x for x,_,_ in minibatches])
        all_y = torch.cat([y for _,y,_ in minibatches])
        all_d = torch.cat([torch.full((x.shape[0], ), i, dtype=torch.int64) for i, (x,_,_) in enumerate(minibatches)])
        len_minibatch = len(all_y)

        if self.cuda:
            all_x = all_x.cuda()
            all_y = all_y.cuda()
            all_d = all_d.cuda()

        all_z = self.featurizer(all_x)

        pred = self.classifier(all_z)
        pred_d = self.discriminator(all_z)

        loss_class = self.loss_type(pred, all_y)
        loss_domain = self.loss_type_d(pred_d, all_d)
        if step % self.d_steps_per_g_step == 0:
            loss = loss_class + loss_domain * self.lambd
        else:
            loss = loss_class

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()/len_minibatch, 
                'loss_class': loss_class.item()/len_minibatch,
                'loss_domain': loss_domain.item()/len_minibatch}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def validate_step(self, loader):
        self.featurizer.eval()
        self.classifier.eval()
        pred_list = []
        acc = 0.0
        loader_len = 0.0

        for batch_idx, (all_x, all_y, _) in enumerate(loader):
            if self.cuda:
                all_x = all_x.cuda()
                all_y = all_y.cuda()

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
            os.remove(checkpoint_path)
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



class IRM(Algorithm):
    def __init__ (self, cfgs, args):
        self.cuda = args.cuda
        self.featurizer = Featurizer(cfgs)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_classes'],
            cfgs['nonlinear_classifier']
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          lr=cfgs['learning_rate'],
                                          weight_decay=cfgs['weight_decay'])
        
        if cfgs['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss()
            self.irm_loss = F.cross_entropy
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")
        
        if self.cuda:
            self.featurizer.cuda()
            self.classifier.cuda()

        self.irm_iter = cfgs['IRM']['iter']
        self.lambd = cfgs['IRM']['lambd'] 

    def _irm_penalty(self, pred, y):
        device = 'cuda' if self.cuda else 'cpu'
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = self.irm_loss(pred[::2] * scale, y[::2])
        loss_2 = self.irm_loss(pred[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, step, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()

        all_x = torch.cat([x for x,_,_ in minibatches])
        all_y = torch.cat([y for _,y,_ in minibatches])
        len_minibatch = len(all_y)

        if self.cuda:
            all_x = all_x.cuda()
            all_y = all_y.cuda()

        pred = self.predict(all_x)

        loss_class = self.loss_type(pred, all_y)
        running_idx = 0 # this is to seperate predictions of each domain, if even batchsize is used for all domain, this can should be handled cleaner
        irm_loss = 0
        if step >= self.irm_iter:
            for i, (x,y,_) in enumerate(minibatches):
                d_pred = pred[running_idx:running_idx + x.shape[0]]
                running_idx += x.shape[0]
                irm_loss += self._irm_penalty(d_pred,y)
            irm_loss /= len(minibatches)

        loss = loss_class + self.lambd * irm_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()/len_minibatch,
                'loss_class': loss_class.item()/len_minibatch,
                'loss_irm': irm_loss.item()/len_minibatch}

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        self.featurizer.eval()
        self.classifier.eval()
        acc = 0.0
        loader_len = 0.0

        pred_list = []

        for batch_idx, (all_x, all_y, _) in enumerate(loader):
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















