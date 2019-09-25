import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer

bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
class_criterion = nn.CrossEntropyLoss().cuda()
def mixup_criterion(y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()

class Trainer(object):
    def __init__(self, opt, model, ema= True):
        self.opt = opt
        self.ema = ema
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.criterion.cuda()
        if  self.ema == True:
            self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        self.model.reset()
        if self.ema == True:
            self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def update(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        #self.model.train()
        #self.optimizer.zero_grad()

        logits= self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
        #import pdb; pdb.set_trace()
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
        
        #loss.backward()
        #self.optimizer.step()
        return loss
    
    
    def update_soft_augmented_mix_nodes(self, inputs, target,target_discrete, idx, idx_unlabeled, adj, opt, mixup_layer):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()
            idx_unlabeled = idx_unlabeled.cuda()

        #self.model.train()
        #self.optimizer.zero_grad()
        
        ### get the loss by mixing the labeled samples  ####
        logits, target_out, idx = self.model.forward_mix(inputs, target, target_discrete, idx,  opt, mixup_layer)
        logits = torch.log_softmax(logits, dim=-1)
        #import pdb; pdb.set_trace()
        loss = -torch.mean(torch.sum(target_out[idx] * logits[idx], dim=-1))
        
        ## reset the adj matrix to original adj matrix##
        self.model.m1.adj = adj
        self.model.m2.adj = adj

        #### get the loss by mixing unlabeled nodes###
        logits, target_out, idx = self.model.forward_mix(inputs, target, target_discrete, idx_unlabeled, opt, mixup_layer)
        logits = torch.log_softmax(logits, dim=-1)
        #import pdb; pdb.set_trace()
        loss_usup = -torch.mean(torch.sum(target_out[idx] * logits[idx], dim=-1))
            
        #loss.backward()
        #self.optimizer.step()
        return loss, loss_usup
    
    
    def update_soft_aux(self, inputs, target,target_discrete, idx, idx_unlabeled, adj, opt, mixup_layer):
        """uses the auxiliary loss as well, which does not use the adjacency information"""
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()
            idx_unlabeled = idx_unlabeled.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        #import pdb ;pdb.set_trace()
        mixup = True
        if mixup == True:
            # get the supervised mixup loss #
            logits, target_a, target_b, lam = self.model.forward_aux(inputs, target=target, train_idx= idx, mixup_input=False, mixup_hidden = True, mixup_alpha = opt['mixup_alpha'],layer_mix=mixup_layer)
            #import pdb; pdb.set_trace()
            mixed_target = lam*target_a + (1-lam)*target_b
            #logits = torch.log_softmax(logits, dim=-1)
            #loss_aux = -(torch.mean(lam*torch.sum(target_a * logits[idx], dim=-1, keepdim= True))+ torch.mean((1-lam)*torch.sum(target_b * logits[idx], dim=-1, keepdim =True)))
            loss = bce_loss(softmax(logits[idx]), mixed_target)

            # get the unsupervised mixup loss #
            logits, target_a, target_b, lam = self.model.forward_aux(inputs, target=target, train_idx= idx_unlabeled, mixup_input=False, mixup_hidden = True, mixup_alpha = opt['mixup_alpha'],layer_mix= mixup_layer)
            mixed_target = lam*target_a + (1-lam)*target_b
            loss_usup = bce_loss(softmax(logits[idx_unlabeled]), mixed_target)
        else:
            logits = self.model.forward_aux(inputs, target=None, train_idx= idx, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
            

            logits = self.model.forward_aux(inputs, target=None, train_idx= idx_unlabeled, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None)
            logits = torch.log_softmax(logits, dim=-1)
            loss_usup = -torch.mean(torch.sum(target[idx_unlabeled] * logits[idx_unlabeled], dim=-1))
        
        return loss, loss_usup

    def update_soft_mix(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()
        #import pdb; pdb.set_trace()
        mixup = False
        if mixup == True:
            logits, target_a, target_b, lam = self.model(inputs, target=target, train_idx= idx, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -(torch.mean(lam*torch.sum(target_a * logits[idx], dim=-1, keepdim= True))+ torch.mean((1-lam)*torch.sum(target_b * logits[idx], dim=-1, keepdim =True)))
        else:
            logits = self.model(inputs, target=None, train_idx= idx, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
        
        """
        layer = random.randint(1,3)
        if layer ==1:
            logits_aux, target_a_aux, target_b_aux, lam_aux = self.model.get_m1_mix(inputs,target= target, train_idx= idx, mixup_alpha = 1.0)
        elif layer ==2:
            logits_aux, target_a_aux, target_b_aux, lam_aux = self.model.get_m2_mix(inputs,target= target, train_idx= idx, mixup_alpha = 1.0)
        elif layer ==3:
            logits_aux, target_a_aux, target_b_aux, lam_aux = self.model.get_m3_mix(inputs,target= target, train_idx= idx, mixup_alpha = 1.0)
        logits_aux = torch.log_softmax(logits_aux, dim=-1)
    
        loss_aux = -(torch.mean(lam_aux*torch.sum(target_a_aux * logits_aux[idx], dim=-1, keepdim= True))+ torch.mean((1-lam_aux)*torch.sum(target_b_aux * logits_aux[idx], dim=-1, keepdim =True)))
       """ 

        loss = loss#+ 1.0*loss_aux
        
        """
        temp = random.randint(0,1)
        if temp==0:
            loss = loss
        else:
            loss = loss_aux
        """
        """
        temp = random.randint(0,1)
        if temp==0:
            loss = loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            optimizer_aux = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])
            layer = random.randint(1,3)
            for i in range(100):
                #import pdb; pdb.set_trace()
                #self.model.zero_grad()
                rand_idx = torch.randint(0, idx.shape[0]-1, (32,))
                
                if layer ==1:
                    logits_aux, target_a_aux, target_b_aux, lam_aux = self.model.get_m1_mix(inputs,target= target, train_idx= idx, mixup_alpha = 1.0)
                elif layer ==2:
                    logits_aux, target_a_aux, target_b_aux, lam_aux = self.model.get_m2_mix(inputs,target= target, train_idx= idx, mixup_alpha = 1.0)
                elif layer ==3:
                    logits_aux, target_a_aux, target_b_aux, lam_aux = self.model.get_m3_mix(inputs,target= target, train_idx= idx, mixup_alpha = 1.0)
                
                logits_aux = torch.log_softmax(logits_aux, dim=-1)
                
                rand_idx = torch.randint(0, idx.shape[0]-1, (32,))
                loss_aux = -(torch.mean(lam_aux*torch.sum(target_a_aux[rand_idx] * logits_aux[idx][rand_idx], dim=-1, keepdim= True))+ torch.mean((1-lam_aux)*torch.sum(target_b_aux[rand_idx] * logits_aux[idx][rand_idx], dim=-1, keepdim =True)))
                loss_aux.backward()
                optimizer_aux.step()
                #print (loss_aux)
        """

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft_mlp(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()
        temp = inputs[idx]
        target_temp = target[idx]
       
        rand_idx = torch.randint(0, idx.shape[0]-1, (32,))
        #import pdb; pdb.set_trace()
        logits, mixed_target = self.model(temp[rand_idx], target=target_temp[rand_idx], mixup_input = False, mixup_hidden= True, mixup_alpha=1.0,layer_mix=4)
        #logits = torch.log_softmax(logits, dim=-1)
        
        #loss = -torch.mean(torch.sum(t * logits, dim=-1))
    
        #loss_func = mixup_criterion(y_a, y_b, lam)
        #class_loss = loss_func(class_criterion, output_mixed_l)
        loss = bce_loss(softmax(logits), mixed_target)

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.eval()
        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)

        return loss.item(), preds, accuracy.item()

    def predict(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits


    def predict_aux(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model.forward_aux(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits

    def predict_noisy(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        #self.model.eval()

        logits = self.model(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits


    def predict_noisy_aux(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        #self.model.eval()

        logits = self.model.forward_aux(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits
    
    
    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
