import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os
from tqdm import tqdm
from utils.networks_utils import Featurizer, Classifier, GRL, ParamDict
from sklearn.cluster import MiniBatchKMeans
import json
import time
import copy


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

    def train(self, num_steps, train_loader, in_val_loader, out_val_loader, test_loader, val_freq, ckpt_freq, i_test_dom, results_dir=None, ckpts_dir=None, cur_step=0):
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
                # _, train_acc = self.validate_step(in_val_loader)
                # _, val_acc = self.validate_step(out_val_loader)
                # _, test_acc = self.validate_step(test_loader)


                _, tr_all_acc, tr_avg_acc = self.validate_step(in_val_loader)
                for i_dom, acc in enumerate(tr_all_acc):
                    if i_dom == i_test_dom:
                        continue
                    loss_list[-1].update({f'tr_dom{i_dom}_acc': acc})

                _, val_all_acc, val_avg_acc = self.validate_step(out_val_loader)
                for i_dom, acc in enumerate(val_all_acc):
                    if i_dom == i_test_dom:
                        continue
                    loss_list[-1].update({f'val_dom{i_dom}_acc': acc})

                _, _, te_acc = self.validate_step(test_loader)

                loss_list[-1].update({f'tr_avg_acc': tr_avg_acc})
                loss_list[-1].update({f'val_avg_acc': val_avg_acc})
                loss_list[-1].update({f'te_dom{i_test_dom}_acc': te_acc})

                mem_gb = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
                
                loss_list[-1].update({'step': float(step),
                                      'mem_gb': mem_gb})
                # loss_list[-1].update({'train_acc': train_acc,
                #                     'val_acc': val_acc,
                #                     'test_acc': test_acc,
                #                     'step': float(step),
                #                     'mem_gb': mem_gb})

                '''
                    Print and save validation results at every val step
                '''
                for key in loss_list[-1].keys():
                    tqdm.write(f"{key}".ljust(15), end = "")
                tqdm.write("")

                for key in loss_list[-1].keys():
                    tqdm.write(f"{loss_list[-1][key]:<10f}     ", end="")
                tqdm.write("\n\n")


            '''
                Save the checkpoints
            '''
            if step+1 % ckpt_freq == 0 or step==total_step-1:
                self.save_ckpt(step, ckpts_dir)
            
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

    def validate_step(self, loader, test_dom=None):
        '''
            Perform validation over the entire loader.
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

        self.n_domains = cfgs['num_domains']

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

        device = 'cuda' if self.cuda else 'cpu'
        all_x = all_x.to(device, non_blocking=True)
        all_y = all_y.to(device, non_blocking=True)

        loss_class = self.loss_type(self.predict(all_x), all_y)
        
        self.optimizer.zero_grad()
        loss_class.backward()
        self.optimizer.step()

        return {'loss_class' : loss_class.item()}

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        device = 'cuda' if self.cuda else 'cpu'
        self.featurizer.eval()
        self.classifier.eval()
        with torch.inference_mode():
            acc = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            loader_len = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            pred_list = []

            for batch_idx, (all_x, all_y, all_d) in enumerate(loader):
                all_x = all_x.to(device, non_blocking=True)
                all_y = all_y.to(device, non_blocking=True)
                all_d = all_d.to(device, non_blocking=True)

                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                
                corrects = torch.eq(pred, all_y).to(dtype=torch.int64)
                acc += torch.bincount(all_d.long(), weights=corrects, minlength=self.n_domains)
                loader_len += torch.bincount(all_d.long(), minlength=self.n_domains)
                pred_list.append(zip(pred.cpu().numpy(),all_y.cpu().numpy()))


        self.featurizer.train()
        self.classifier.train()
        
        avg_acc = sum(acc) / sum(loader_len)
        
        loader_len = torch.clamp(loader_len, min=1)
        all_acc = acc / loader_len

        return pred_list, all_acc.cpu().numpy().tolist(), avg_acc.cpu().numpy().item()


    def save_ckpt(self, step, ckpts_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(ckpts_dir, f'Best_ckpt.pth.rar')
        else:
            checkpoint_path = os.path.join(ckpts_dir, f'Step_{step}_ckpt.pth.rar')

        state_dict = {
            'step': step,
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
        step = state_dict['step']
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return step

class Proposed1(Algorithm):
    def __init__ (self, cfgs, args):
        self.cuda = args.cuda
        self.featurizer = Featurizer(cfgs)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_classes'],
            cfgs['nonlinear_classifier']
        )

        self.d_classifier = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_train_domains'],
            cfgs['Proposed1']['d_nonlinear_classifier']
        )

        self.C_branch = nn.Sequential(nn.Flatten(),
                                      nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
                                      nn.ReLU(),
                                      nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
                                      )

        self.D_branch = nn.Sequential(nn.Flatten(),
                                      nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
                                      nn.ReLU(),
                                      nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
                                      )

        self.n_domains = cfgs['num_domains']
        self.iter = cfgs['Proposed1']['iter']

        self.network = nn.Sequential(self.featurizer, self.C_branch, self.classifier)
        self.optimizer = torch.optim.Adam(list(self.network.parameters()) + 
                                          list(self.featurizer.parameters()) +
                                          list(self.D_branch.parameters()) +
                                          list(self.d_classifier.parameters()), 
                                          lr=cfgs['learning_rate'],
                                          weight_decay=cfgs['weight_decay'])    
        
        if cfgs['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")

        if cfgs['Proposed1']['loss_type_d'] == 'CrossEntropy':
            self.loss_type_d = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['Proposed1']['loss_type_d']} is not implemented")

        if self.cuda:
            self.featurizer.cuda()
            self.C_branch.cuda()
            self.classifier.cuda()
            self.d_classifier.cuda()
            self.D_branch.cuda()

    def update(self, minibatches, step, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()

        all_x = torch.cat([x for x,_,_ in minibatches])
        all_y = torch.cat([y for _,y,_ in minibatches])
        all_d = torch.cat([torch.full((x.shape[0], ), i, dtype=torch.int64) for i, (x,_,_) in enumerate(minibatches)])

        device = 'cuda' if self.cuda else 'cpu'
        all_x = all_x.to(device, non_blocking=True)
        all_y = all_y.to(device, non_blocking=True)
        all_d = all_d.to(device, non_blocking=True)

        all_z = self.featurizer(all_x)

        lambd = 1 if step >= self.iter else 0

        pred = self.classifier(self.C_branch(all_z) - lambd * self.D_branch(all_z).detach())
        pred_d = self.d_classifier(self.D_branch(all_z))

        loss_class = self.loss_type(pred, all_y)
        loss_domain = self.loss_type_d(pred_d, all_d)

        loss = loss_class + loss_domain

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 
                'loss_class': loss_class.item(),
                'loss_domain': loss_domain.item()}

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        device = 'cuda' if self.cuda else 'cpu'
        self.featurizer.eval()
        self.classifier.eval()
        with torch.inference_mode():
            acc = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            loader_len = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            pred_list = []

            for batch_idx, (all_x, all_y, all_d) in enumerate(loader):
                all_x = all_x.to(device, non_blocking=True)
                all_y = all_y.to(device, non_blocking=True)
                all_d = all_d.to(device, non_blocking=True)

                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                
                corrects = torch.eq(pred, all_y).to(dtype=torch.int64)
                acc += torch.bincount(all_d.long(), weights=corrects, minlength=self.n_domains)
                loader_len += torch.bincount(all_d.long(), minlength=self.n_domains)
                pred_list.append(zip(pred.cpu().numpy(),all_y.cpu().numpy()))

        self.featurizer.train()
        self.classifier.train()
        
        avg_acc = sum(acc) / sum(loader_len)
        
        loader_len = torch.clamp(loader_len, min=1)
        all_acc = acc / loader_len

        return pred_list, all_acc.cpu().numpy().tolist(), avg_acc.cpu().numpy().item()


    def save_ckpt(self, step, ckpts_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(ckpts_dir, f'Best_ckpt.pth.rar')
        else:
            checkpoint_path = os.path.join(ckpts_dir, f'Step_{step}_ckpt.pth.rar')

        state_dict = {
            'step': step,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'np_random': np.random.get_state(),
            'd_branch': self.D_branch.state_dict(),
            'd_classifier': self.d_classifier.state_dict()
        }
        if torch.cuda.is_available():
            state_dict.update({'cuda_rng': torch.cuda.get_rng_state()})
        torch.save(state_dict, checkpoint_path)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=False)
        step = state_dict['step']
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return step    


class Proposed2(Algorithm):
    def __init__ (self, cfgs, args):
        self.cuda = args.cuda
        self.featurizer = Featurizer(cfgs)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_classes'],
            is_nonlinear=False
        )

        self.d_classifier = Classifier(
            self.featurizer.n_outputs,
            # cfgs['num_train_domains'],
            cfgs['Proposed2']['num_clusters'],
            cfgs['Proposed2']['d_nonlinear_classifier']
        )

        self.n_domains = cfgs['num_domains']
        self.lambd_orth = cfgs['Proposed2']['lambd_orth']
        self.lambd_domain = cfgs['Proposed2']['lambd_domain']
        self.lambd_cross_sample = cfgs['Proposed2']['lambd_cross']
        # P2
        # self.inner_steps = cfgs['Proposed2']['inner_steps']
        self.n_train_domains = cfgs['Proposed2']['num_clusters']
        self.theta = cfgs['Proposed2']['theta']

        self.mbk = MiniBatchKMeans(
            n_clusters=self.n_train_domains,
            batch_size=16*6,
            reassignment_ratio=cfgs['Proposed2']['reassign_ratio'],
            random_state=0,
            n_init=3,
        )

        self.CateRelated = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
            nn.ReLU(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs)
        )

        self.EnvRelated = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
            nn.ReLU(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs)
        )

        self.ClassPrototype = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.classifier[-1].weight.shape[1],self.featurizer.n_outputs), # classifier.weight have shape (n_class, n_hidden)
            nn.ReLU(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
        )


        # P2
        # self.classifier_per_d = Classifier(self.featurizer.n_outputs,
        #                                    cfgs['num_classes'],
        #                                    is_nonlinear=cfgs['nonlinear_classifier']) 

        self.network = nn.Sequential(self.featurizer, self.CateRelated, self.classifier)
        self.optimizer = torch.optim.Adam(list(self.network.parameters())+
                                          list(self.EnvRelated.parameters())+
                                          list(self.d_classifier.parameters())+
                                          list(self.ClassPrototype.parameters()), 
                                          lr=cfgs['learning_rate'],
                                          weight_decay=cfgs['weight_decay'])

        # P2
        # self.optimizer_inner = torch.optim.Adam([param for cl in self.classifier_per_d for param in cl.parameters()], 
        #                                         lr=cfgs['learning_rate'],
        #                                         weight_decay=cfgs['weight_decay'])        
        
        if cfgs['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")

        if cfgs['CFSM']['loss_type_d'] == 'CrossEntropy':
            self.d_loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type_d']} is not implemented")

        if self.cuda:
            self.featurizer.cuda()
            self.classifier.cuda()
            self.CateRelated.cuda()
            self.EnvRelated.cuda()
            self.d_classifier.cuda()
            self.ClassPrototype.cuda()
            # P2
            # for classifier in self.classifier_per_d:
            #     classifier.cuda()

    def orth_loss(self):
        product = torch.inner(self.CateRelated[1].weight, self.EnvRelated[1].weight)
        return (product ** 2).sum()

    @torch.no_grad()
    def get_pseudo_label(self, z_env):
        z = z_env.detach().cpu().numpy()
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
        self.mbk.partial_fit(z)

        pseudo_labels = self.mbk.predict(z)
        return torch.from_numpy(pseudo_labels).long().to(device=z_env.device)

    def cross_sample_loss(self, z_cate, all_y, all_d):
        z_cate_norm = nn.functional.normalize(z_cate, p=2, dim=1)
        cos_sim = torch.inner(z_cate_norm, z_cate_norm)
        self_pair = torch.eye(len(all_y), dtype=torch.bool, device=all_y.device)
        pos_pair = (all_y.unsqueeze(0) == all_y.unsqueeze(1)) & (~self_pair)

        in_dom_pair = (all_d.unsqueeze(0) == all_d.unsqueeze(1)) & (~self_pair)

        if not pos_pair.any():
            return torch.tensor(0, device=all_y.device)
        
        threshold = torch.mean(cos_sim[pos_pair]) * self.theta
        neg_pair = (cos_sim > threshold) & (~pos_pair) & (~self_pair) & in_dom_pair

        if not neg_pair.any():
            return torch.tensor(0, device=all_y.device)
        
        idx_i, idx_j = torch.where(neg_pair)
        z_pos = z_cate_norm[idx_i]
        z_neg = z_cate_norm[idx_j]
        y_pos = all_y[idx_i]

        prototype = nn.functional.normalize(self.ClassPrototype((self.classifier[-1].weight)), p=2, dim=1)

        return torch.mean(torch.sum(z_neg * prototype[y_pos], dim=1) - torch.sum(z_pos * prototype[y_pos], dim=1))

    def update(self, minibatches, step, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()
        self.CateRelated.train()

        all_x = torch.cat([x for x,_,_ in minibatches])
        all_y = torch.cat([y for _,y,_ in minibatches])
        all_d = torch.cat([torch.full((x.shape[0], ), i, dtype=torch.int64) for i, (x,_,_) in enumerate(minibatches)])

        device = 'cuda' if self.cuda else 'cpu'
        all_x = all_x.to(device, non_blocking=True)
        all_y = all_y.to(device, non_blocking=True)
        all_d = all_d.to(device, non_blocking=True)

        all_z = self.featurizer(all_x)
        z_cate = self.CateRelated(all_z)
        z_env = self.EnvRelated(all_z)

        all_d = self.get_pseudo_label(z_env)

        # P2
        # for step in range(self.inner_steps):
            #     loss_class_inner = self.loss_type(self.classifier_per_d(self.CateRelated(self.featurizer(x))), y)
            #     self.optimizer_inner.zero_grad()
            #     loss_class_inner.backward()
            #     self.optimizer_inner.step()
        #     if i_dom == 0:
        #         inner_weight = ParamDict(copy.deepcopy(self.classifier_per_d.state_dict()))
        #     else:
        #         inner_weight += ParamDict(copy.deepcopy(self.classifier_per_d.state_dict()))
        # self.classifier.load_state_dict(inner_weight/len(minibatches))

        pred = self.classifier(z_cate)
        d_pred = self.d_classifier(z_env)

        loss_class = self.loss_type(pred, all_y)
        loss_domain = self.d_loss_type(d_pred, all_d)
        loss_orth = self.orth_loss()
        loss_cross_sample = self.cross_sample_loss(z_cate, all_y, all_d)


        loss = loss_class + self.lambd_domain * loss_domain + self.lambd_orth * loss_orth + self.lambd_cross_sample * loss_cross_sample

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss'          : loss.item(),
                'loss_class'    : loss_class.item(),
                'loss_domain'   : loss_domain.item(),
                'loss_orth'     : loss_orth.item(),
                'loss_cross_sample' : loss_cross_sample.item(),
                }

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        device = 'cuda' if self.cuda else 'cpu'
        self.featurizer.eval()
        self.CateRelated.eval()
        self.classifier.eval()
        with torch.inference_mode():
            acc = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            loader_len = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            pred_list = []

            for batch_idx, (all_x, all_y, all_d) in enumerate(loader):
                all_x = all_x.to(device, non_blocking=True)
                all_y = all_y.to(device, non_blocking=True)
                all_d = all_d.to(device, non_blocking=True)

                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                
                corrects = torch.eq(pred, all_y).to(dtype=torch.int64)
                acc += torch.bincount(all_d.long(), weights=corrects, minlength=self.n_domains)
                loader_len += torch.bincount(all_d.long(), minlength=self.n_domains)
                pred_list.append(zip(pred.cpu().numpy(),all_y.cpu().numpy()))


        self.featurizer.train()
        self.classifier.train()
        
        avg_acc = sum(acc) / sum(loader_len)
        
        loader_len = torch.clamp(loader_len, min=1)
        all_acc = acc / loader_len

        return pred_list, all_acc.cpu().numpy().tolist(), avg_acc.cpu().numpy().item()


    def save_ckpt(self, step, ckpts_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(ckpts_dir, f'Best_ckpt.pth.rar')
        else:
            checkpoint_path = os.path.join(ckpts_dir, f'Step_{step}_ckpt.pth.rar')

        state_dict = {
            'step': step,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'np_random': np.random.get_state(),
            'EnvRelated': self.EnvRelated.state_dict(),
            'd_classifier': self.d_classifier.state_dict(),
        }
        if torch.cuda.is_available():
            state_dict.update({'cuda_rng': torch.cuda.get_rng_state()})
        torch.save(state_dict, checkpoint_path)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=False)
        step = state_dict['step']
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return step



class DANN(Algorithm):
    def __init__ (self, cfgs, args):
        self.cuda = args.cuda
        self.lambd = cfgs['DANN']['lambd']
        self.lambd_iter = cfgs['DANN']['lambd_iter']
        # self.d_steps_per_g_step = cfgs['DANN']['d_steps_per_g_step']
        self.featurizer = Featurizer(cfgs)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_classes'],
            cfgs['nonlinear_classifier']
        )

        self.discriminator = nn.Sequential(
            GRL(lambd=0.0),
            Classifier(
            self.featurizer.n_outputs,
            cfgs['num_train_domains'],
            cfgs['DANN']['nonlinear_discriminator']
            )
        )

        self.n_domains = cfgs['num_domains']

        self.optimizer = torch.optim.Adam((list(self.featurizer.parameters())+
                                           list(self.classifier.parameters())+
                                           list(self.discriminator.parameters())), 
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


        device = 'cuda' if self.cuda else 'cpu'
        all_x = all_x.to(device, non_blocking=True)
        all_y = all_y.to(device, non_blocking=True)
        all_d = all_d.to(device, non_blocking=True)

        running_lambd = self.lambd * (2/(1+np.exp((-step/self.lambd_iter)))-1)
        self.discriminator[0] = GRL(lambd=running_lambd)     

        all_z = self.featurizer(all_x)

        pred = self.classifier(all_z)
        pred_d = self.discriminator(all_z)

        loss_class = self.loss_type(pred, all_y)
        loss_domain = self.loss_type_d(pred_d, all_d)

        loss = loss_class + loss_domain

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 
                'loss_class': loss_class.item(),
                'loss_domain': loss_domain.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def validate_step(self, loader):
        device = 'cuda' if self.cuda else 'cpu'
        self.featurizer.eval()
        self.classifier.eval()
        with torch.inference_mode():
            acc = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            loader_len = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            pred_list = []

            for batch_idx, (all_x, all_y, all_d) in enumerate(loader):
                all_x = all_x.to(device, non_blocking=True)
                all_y = all_y.to(device, non_blocking=True)
                all_d = all_d.to(device, non_blocking=True)

                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                
                corrects = torch.eq(pred, all_y).to(dtype=torch.int64)
                acc += torch.bincount(all_d.long(), weights=corrects, minlength=self.n_domains)
                loader_len += torch.bincount(all_d.long(), minlength=self.n_domains)
                pred_list.append(zip(pred.cpu().numpy(),all_y.cpu().numpy()))


        self.featurizer.train()
        self.classifier.train()
        
        avg_acc = sum(acc) / sum(loader_len)
        
        loader_len = torch.clamp(loader_len, min=1)
        all_acc = acc / loader_len

        return pred_list, all_acc.cpu().numpy().tolist(), avg_acc.cpu().numpy().item()


    def save_ckpt(self, step, ckpts_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(ckpts_dir, f'Best_ckpt.pth.rar')
            os.remove(checkpoint_path)
        else:
            checkpoint_path = os.path.join(ckpts_dir, f'Step_{step}_ckpt.pth.rar')

        state_dict = {
            'step': step,
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
        step = state_dict['step']
        self.featurizer.load_state_dict(state_dict['featurizer'])
        self.classifier.load_state_dict(state_dict['classifier'])
        self.discriminator.load_state_dict(state_dict['discriminator'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return step

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
        self.lr = cfgs['learning_rate']
        self.wd = cfgs['weight_decay']
    
        self.n_domains = cfgs['num_domains']

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

        device = 'cuda' if self.cuda else 'cpu'
        all_x = all_x.to(device, non_blocking=True)
        all_y = all_y.to(device, non_blocking=True)

        pred = self.predict(all_x)

        loss_class = self.loss_type(pred, all_y)
        running_idx = 0 # this is to seperate predictions of each domain, if even batchsize is used for all domain, this can be handled better
        irm_loss = torch.tensor(0.0, requires_grad=False).to(device)
        if step >= self.irm_iter:
            for i_dom, (x,_,_) in enumerate(minibatches):
                d_pred = pred[running_idx:running_idx + x.shape[0]]
                d_y = all_y[running_idx:running_idx + x.shape[0]]
                running_idx += x.shape[0]
                irm_loss += self._irm_penalty(d_pred,d_y)
            irm_loss /= all_x.shape[0]

        if step == self.irm_iter:
            self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                              lr=self.lr,
                                              weight_decay=self.wd)

        loss = loss_class + self.lambd * irm_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(),
                'loss_class': loss_class.item(),
                'loss_irm': irm_loss.item()}

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        device = 'cuda' if self.cuda else 'cpu'
        self.featurizer.eval()
        self.classifier.eval()
        with torch.inference_mode():
            acc = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            loader_len = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            pred_list = []

            for batch_idx, (all_x, all_y, all_d) in enumerate(loader):
                all_x = all_x.to(device, non_blocking=True)
                all_y = all_y.to(device, non_blocking=True)
                all_d = all_d.to(device, non_blocking=True)

                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                
                corrects = torch.eq(pred, all_y).to(dtype=torch.int64)
                acc += torch.bincount(all_d.long(), weights=corrects, minlength=self.n_domains)
                loader_len += torch.bincount(all_d.long(), minlength=self.n_domains)
                pred_list.append(zip(pred.cpu().numpy(),all_y.cpu().numpy()))


        self.featurizer.train()
        self.classifier.train()
        
        avg_acc = sum(acc) / sum(loader_len)
        
        loader_len = torch.clamp(loader_len, min=1)
        all_acc = acc / loader_len

        return pred_list, all_acc.cpu().numpy().tolist(), avg_acc.cpu().numpy().item()


    def save_ckpt(self, step, ckpts_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(ckpts_dir, f'Best_ckpt.pth.rar')
        else:
            checkpoint_path = os.path.join(ckpts_dir, f'Step_{step}_ckpt.pth.rar')

        state_dict = {
            'step': step,
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
        step = state_dict['step']
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return step

class VRex(Algorithm):
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
        self.lr = cfgs['learning_rate']
        self.wd = cfgs['weight_decay']

        self.n_domains = cfgs['num_domains']

        if cfgs['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")
        
        self.lambd = cfgs['VRex']['lambd']
        self.iter = cfgs['VRex']['iter']
        
        if self.cuda:
            self.featurizer.cuda()
            self.classifier.cuda()

    def update(self, minibatches, step, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()

        all_x = torch.cat([x for x,_,_ in minibatches])
        all_y = torch.cat([y for _,y,_ in minibatches])

        device = 'cuda' if self.cuda else 'cpu'
        all_x = all_x.to(device, non_blocking=True)
        all_y = all_y.to(device, non_blocking=True)

        pred = self.predict(all_x)
        running_idx = 0
        losses = torch.zeros(len(minibatches))

        for i, (x, y, _) in enumerate(minibatches):
            d_pred = pred[running_idx:running_idx + x.shape[0]]
            d_y = all_y[running_idx:running_idx + x.shape[0]]
            running_idx += x.shape[0]
            d_loss = self.loss_type(d_pred, d_y)
            losses[i] = d_loss

        if step >= self.iter:
            penalty_weight = self.lambd
        else:
            penalty_weight = 0.0

        if step == self.iter:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.lr,
                weight_decay=self.wd)

        loss_class = losses.mean()
        loss_vrex = ((losses - loss_class) ** 2).mean()
        loss = loss_class + penalty_weight * loss_vrex

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 
                'loss_class' : loss_class.item(),
                'loss_vrex': loss_vrex.item()}

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        device = 'cuda' if self.cuda else 'cpu'
        self.featurizer.eval()
        self.classifier.eval()
        with torch.inference_mode():
            acc = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            loader_len = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            pred_list = []

            for batch_idx, (all_x, all_y, all_d) in enumerate(loader):
                all_x = all_x.to(device, non_blocking=True)
                all_y = all_y.to(device, non_blocking=True)
                all_d = all_d.to(device, non_blocking=True)

                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                
                corrects = torch.eq(pred, all_y).to(dtype=torch.int64)
                acc += torch.bincount(all_d.long(), weights=corrects, minlength=self.n_domains)
                loader_len += torch.bincount(all_d.long(), minlength=self.n_domains)
                pred_list.append(zip(pred.cpu().numpy(),all_y.cpu().numpy()))


        self.featurizer.train()
        self.classifier.train()
        
        avg_acc = sum(acc) / sum(loader_len)
        
        loader_len = torch.clamp(loader_len, min=1)
        all_acc = acc / loader_len

        return pred_list, all_acc.cpu().numpy().tolist(), avg_acc.cpu().numpy().item()


    def save_ckpt(self, step, ckpts_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(ckpts_dir, f'Best_ckpt.pth.rar')
        else:
            checkpoint_path = os.path.join(ckpts_dir, f'Step_{step}_ckpt.pth.rar')

        state_dict = {
            'step': step,
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
        step = state_dict['step']
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return step

class Fish(Algorithm):
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

        self.n_domains = cfgs['num_domains']

        if cfgs['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")

        self.featurizer_inner = Featurizer(cfgs)
        self.classifier_inner = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_classes'],
            cfgs['nonlinear_classifier']
        )

        self.network_inner = nn.Sequential(self.featurizer_inner, self.classifier_inner)
        self.optimizer_inner = torch.optim.Adam(self.network_inner.parameters(), 
                                          lr=cfgs['learning_rate'],
                                          weight_decay=cfgs['weight_decay'])

        self.lr_meta = cfgs['Fish']['lr_meta']

        if self.cuda:
            self.network.cuda()
            self.network_inner.cuda()

    def update(self, minibatches, step, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()

        self.featurizer_inner.train()
        self.classifier_inner.train()

        loss_class = 0.0

        device = 'cuda' if self.cuda else 'cpu'
        self.network_inner.load_state_dict(self.network.state_dict())
        for x,y,_ in minibatches:
            x = x.to(device)
            y = y.to(device)
            loss_class_inner = self.loss_type(self.network_inner(x), y)
            loss_class += loss_class_inner.item()

            self.optimizer_inner.zero_grad()
            loss_class_inner.backward()
            self.optimizer_inner.step()

        meta_weights = ParamDict(self.network.state_dict())
        inner_weights = ParamDict(self.network_inner.state_dict())
        meta_weights += self.lr_meta * (inner_weights - meta_weights)
        self.network.load_state_dict(meta_weights)

        return {'loss_class' : loss_class/len(minibatches)}

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        device = 'cuda' if self.cuda else 'cpu'
        self.featurizer.eval()
        self.classifier.eval()
        with torch.inference_mode():
            acc = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            loader_len = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            pred_list = []

            for batch_idx, (all_x, all_y, all_d) in enumerate(loader):
                all_x = all_x.to(device, non_blocking=True)
                all_y = all_y.to(device, non_blocking=True)
                all_d = all_d.to(device, non_blocking=True)

                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                
                corrects = torch.eq(pred, all_y).to(dtype=torch.int64)
                acc += torch.bincount(all_d.long(), weights=corrects, minlength=self.n_domains)
                loader_len += torch.bincount(all_d.long(), minlength=self.n_domains)
                pred_list.append(zip(pred.cpu().numpy(),all_y.cpu().numpy()))


        self.featurizer.train()
        self.classifier.train()
        
        avg_acc = sum(acc) / sum(loader_len)
        
        loader_len = torch.clamp(loader_len, min=1)
        all_acc = acc / loader_len

        return pred_list, all_acc.cpu().numpy().tolist(), avg_acc.cpu().numpy().item()


    def save_ckpt(self, step, ckpts_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(ckpts_dir, f'Best_ckpt.pth.rar')
        else:
            checkpoint_path = os.path.join(ckpts_dir, f'Step_{step}_ckpt.pth.rar')

        state_dict = {
            'step': step,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'network_inner': self.network_inner.state_dict(),
            'optimizer_inner': self.optimizer_inner.state_dict(),
            'rng': torch.get_rng_state(),
            'np_random': np.random.get_state(),
        }
        if torch.cuda.is_available():
            state_dict.update({'cuda_rng': torch.cuda.get_rng_state()})
        torch.save(state_dict, checkpoint_path)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=False)
        step = state_dict['step']
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.network_inner.load_state_dict(state_dict['network_inner']),
        self.optimizer_inner.load_state_dict(state_dict['optimizer_inner']),
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return step

class CFSM(Algorithm):
    def __init__ (self, cfgs, args):
        self.cuda = args.cuda
        self.featurizer = Featurizer(cfgs)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_classes'],
            is_nonlinear=False # Linear classifier is required
        )

        self.d_classifier = Classifier(
            self.featurizer.n_outputs,
            cfgs['num_train_domains'],
            cfgs['CFSM']['d_nonlinear_classifier']
        )

        self.n_domains = cfgs['num_domains']
        self.theta = cfgs['CFSM']['theta']
        self.lambd_orth = cfgs['CFSM']['lambd_orth']
        self.lambd_domain = cfgs['CFSM']['lambd_domain']
        self.lambd_cross_dom = cfgs['CFSM']['lambd_cross_dom']
        self.lambd_cross_sample = cfgs['CFSM']['lambd_cross_sample']

        self.CateRelated = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
            nn.ReLU(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs)
        )

        self.EnvRelated = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
            nn.ReLU(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs)
        )

        self.ClassPrototype = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.classifier[-1].weight.shape[1],self.featurizer.n_outputs), # classifier.weight have shape (n_class, n_hidden)
            nn.ReLU(),
            nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs),
        )

        self.network = nn.Sequential(self.featurizer, self.CateRelated, self.classifier)
        self.optimizer = torch.optim.Adam(list(self.network.parameters())+
                                          list(self.EnvRelated.parameters())+
                                          list(self.d_classifier.parameters())+
                                          list(self.ClassPrototype.parameters()), 
                                          lr=cfgs['learning_rate'],
                                          weight_decay=cfgs['weight_decay'])
        
        if cfgs['loss_type'] == 'CrossEntropy':
            self.loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type']} is not implemented")

        if cfgs['CFSM']['loss_type_d'] == 'CrossEntropy':
            self.d_loss_type = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{cfgs['loss_type_d']} is not implemented")

        if self.cuda:
            self.featurizer.cuda()
            self.classifier.cuda()
            self.CateRelated.cuda()
            self.EnvRelated.cuda()
            self.d_classifier.cuda()
            self.ClassPrototype.cuda()

    def orth_loss(self):
        product = torch.inner(self.CateRelated[1].weight, self.EnvRelated[1].weight)
        return (product ** 2).mean()

    def cross_sample_loss(self, z_cate, all_y, all_d):
        z_cate_norm = nn.functional.normalize(z_cate, p=2, dim=1)
        cos_sim = torch.inner(z_cate_norm, z_cate_norm)
        self_pair = torch.eye(len(all_y), dtype=torch.bool, device=all_y.device)
        pos_pair = (all_y.unsqueeze(0) == all_y.unsqueeze(1)) & (~self_pair)

        in_dom_pair = (all_d.unsqueeze(0) == all_d.unsqueeze(1)) & (~self_pair)

        if not pos_pair.any():
            return torch.tensor(0, device=all_y.device)
        
        threshold = torch.mean(cos_sim[pos_pair]) * self.theta
        neg_pair = (cos_sim > threshold) & (~pos_pair) & (~self_pair) & in_dom_pair

        if not neg_pair.any():
            return torch.tensor(0, device=all_y.device)
        
        idx_i, idx_j = torch.where(neg_pair)
        z_pos = z_cate_norm[idx_i]
        z_neg = z_cate_norm[idx_j]
        y_pos = all_y[idx_i]

        prototype = nn.functional.normalize(self.ClassPrototype((self.classifier[-1].weight)), p=2, dim=1)

        return torch.mean(torch.sum(z_neg * prototype[y_pos], dim=1) - torch.sum(z_pos * prototype[y_pos], dim=1))

    def cross_dom_loss(self, z_cate, all_y, all_d):
        z_cate_norm = nn.functional.normalize(z_cate, p=2, dim=1)
        cos_sim = torch.inner(z_cate_norm, z_cate_norm)
        self_pair = torch.eye(len(all_y), dtype=torch.bool, device=all_y.device)
        in_label_pair = (all_y.unsqueeze(0) == all_y.unsqueeze(1)) & (~self_pair)
        cross_dom_pair = (all_d.unsqueeze(0) != all_d.unsqueeze(1))

        if not in_label_pair.any():
            return torch.tensor(0, device=all_y.device)
        
        # threshold = torch.mean(cos_sim[cross_label_pair])
        neg_pair = (cross_dom_pair) & (in_label_pair) # & (cos_sim < self.threshold)

        if not neg_pair.any():
            return torch.tensor(0, device=all_y.device)

        # idx_i, idx_j = torch.where(neg_pair)
        # z_pos = z_cate[idx_i]
        # z_neg = z_cate[idx_j]

        # for param in self.classifier.parameters():
        #     param.requires_grad = False

        # pred_Zs = self.classifier(z_pos-z_neg)

        # loss_cross_dom = F.cross_entropy(pred_Zs, torch.ones_like(pred_Zs)/pred_Zs.shape[-1])

        # for param in self.classifier.parameters():
        #     param.requires_grad = True

        # return loss_cross_dom
        
        idx_i, idx_j = torch.where(neg_pair)
        z_pos = z_cate[idx_i]
        z_neg = z_cate[idx_j]
        y_pos = all_y[idx_i]

        prototype = nn.functional.normalize(self.ClassPrototype((self.classifier[-1].weight)), p=2, dim=1)

        return torch.norm(torch.sum(z_neg * prototype[y_pos], dim=1) - torch.sum(z_pos * prototype[y_pos], dim=1), p='fro')

    def update(self, minibatches, step, unlabeled=None):
        self.featurizer.train()
        self.classifier.train()
        self.CateRelated.train()

        all_x = torch.cat([x for x,_,_ in minibatches])
        all_y = torch.cat([y for _,y,_ in minibatches])
        all_d = torch.cat([torch.full((x.shape[0], ), i, dtype=torch.int64) for i, (x,_,_) in enumerate(minibatches)])

        device = 'cuda' if self.cuda else 'cpu'
        all_x = all_x.to(device, non_blocking=True)
        all_y = all_y.to(device, non_blocking=True)
        all_d = all_d.to(device, non_blocking=True)

        all_z = self.featurizer(all_x)
        z_cate = self.CateRelated(all_z)
        z_env = self.EnvRelated(all_z)

        pred = self.classifier(z_cate)
        d_pred = self.d_classifier(z_env)

        loss_class = self.loss_type(pred, all_y)
        loss_domain = self.d_loss_type(d_pred, all_d)
        loss_orth = self.orth_loss()
        loss_cross_dom = self.cross_dom_loss(z_cate, all_y, all_d) 
        loss_cross_sample = self.cross_sample_loss(z_cate, all_y, all_d)

        loss = loss_class + self.lambd_domain * loss_domain + self.lambd_orth * loss_orth + self.lambd_cross_sample * loss_cross_sample
        + self.lambd_cross_dom * loss_cross_dom

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss'          : loss.item(),
                'loss_class'    : loss_class.item(),
                'loss_domain'   : loss_domain.item(),
                'loss_orth'     : loss_orth.item(),
                'loss_cross_dom': loss_cross_dom.item(),
                'loss_cross_sa' : loss_cross_sample.item(),
                }

    def predict(self, x):
        return self.network(x)

    def validate_step(self, loader):
        device = 'cuda' if self.cuda else 'cpu'
        self.featurizer.eval()
        self.CateRelated.eval()
        self.classifier.eval()
        with torch.inference_mode():
            acc = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            loader_len = torch.zeros(self.n_domains, dtype=torch.float32, device=device)
            pred_list = []

            for batch_idx, (all_x, all_y, all_d) in enumerate(loader):
                all_x = all_x.to(device, non_blocking=True)
                all_y = all_y.to(device, non_blocking=True)
                all_d = all_d.to(device, non_blocking=True)

                pred = self.predict(all_x)
                _, pred = pred.max(1) # same as np.argmax()
                
                corrects = torch.eq(pred, all_y).to(dtype=torch.int64)
                acc += torch.bincount(all_d.long(), weights=corrects, minlength=self.n_domains)
                loader_len += torch.bincount(all_d.long(), minlength=self.n_domains)
                pred_list.append(zip(pred.cpu().numpy(),all_y.cpu().numpy()))


        self.featurizer.train()
        self.classifier.train()
        self.CateRelated.train()
        
        avg_acc = sum(acc) / sum(loader_len)
        
        loader_len = torch.clamp(loader_len, min=1)
        all_acc = acc / loader_len

        return pred_list, all_acc.cpu().numpy().tolist(), avg_acc.cpu().numpy().item()


    def save_ckpt(self, step, ckpts_dir, is_best=False):
        if is_best:
            checkpoint_path = os.path.join(ckpts_dir, f'Best_ckpt.pth.rar')
        else:
            checkpoint_path = os.path.join(ckpts_dir, f'Step_{step}_ckpt.pth.rar')

        state_dict = {
            'step': step,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'np_random': np.random.get_state(),
            'EnvRelated': self.EnvRelated.state_dict(),
            'd_classifier': self.d_classifier.state_dict(),
            'ClassPrototype': self.ClassPrototype.state_dict(),
        }
        if torch.cuda.is_available():
            state_dict.update({'cuda_rng': torch.cuda.get_rng_state()})
        torch.save(state_dict, checkpoint_path)        

    def load_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=False)
        step = state_dict['step']
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        torch.set_rng_state(state_dict['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict['cuda_rng'])
        np.random.set_state(state_dict['np_random'])
        return step











