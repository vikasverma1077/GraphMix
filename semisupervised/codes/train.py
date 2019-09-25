import sys
import os
import copy
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trainer import Trainer
from gnn import GNNq, GNNp, MLP, GNN_mix
from ramps import *
from losses import *
import loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str, default='exp', help = 'name of the folder where the results are saved')
parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--mixup_alpha', type=float, default=1.0, help='alpha for mixing')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--pre_epoch', type=int, default=200, help='Number of pre-training epochs.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
parser.add_argument('--use_gold', type=int, default=1, help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
parser.add_argument('--draw', type=str, default='max', help='Method for drawing object labels, max for max-pooling, smp for sampling.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
### ict hyperparameters ###
#parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                    metavar='LR', help='max learning rate')
#parser.add_argument('--initial_lr', default=0.0, type=float,##TODO
#                    metavar='LR', help='initial learning rate when using linear rampup')
#parser.add_argument('--lr_rampup', default=0, type=int, metavar='EPOCHS',##TODO
#                    help='length of learning rate rampup in the beginning')
#parser.add_argument('--lr_rampdown_epochs', default=None, type=int, metavar='EPOCHS',##TODO
#                    help='length of learning rate cosine rampdown (>= length of training)')
#parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
#parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
#parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                    help='momentum')
#parser.add_argument('--nesterov', action='store_true',
#                    help='use nesterov momentum')
#parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                    help='ema variable decay rate (default: 0.999)')
#parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
#                    help='use consistency loss with given weight (default: None)')
parser.add_argument('--consistency_type', default="mse", type=str, metavar='TYPE',
                    choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--consistency_rampup_starts', default=30, type=int, metavar='EPOCHS',
                    help='epoch at which consistency loss ramp-up starts')
parser.add_argument('--consistency_rampup_ends', default=30, type=int, metavar='EPOCHS',
                    help='lepoch at which consistency loss ramp-up ends')
#parser.add_argument('--mixup_sup_alpha', default=0.0, type=float,
#                    help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
#parser.add_argument('--mixup_usup_alpha', default=0.0, type=float,
#                    help='for unsupervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
#parser.add_argument('--mixup_hidden', action='store_true',
#                    help='apply mixup in hidden layers')
#parser.add_argument('--num_mix_layer', default=3, type=int,
#                    help='number of hidden layers on which mixup is applied in addition to input layer')
parser.add_argument('--mixup_consistency', default=1.0, type=float,
                    help='max consistency coeff for mixup usup loss')

args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)


    
net_file = opt['dataset'] + '/net.txt'
label_file = opt['dataset'] + '/label.txt'
feature_file = opt['dataset'] + '/feature.txt'
train_file = opt['dataset'] + '/train.txt'
dev_file = opt['dataset'] + '/dev.txt'
test_file = opt['dataset'] + '/test.txt'

#import pdb;pdb.set_trace()
### create a temporart net file
exp_dir = os.path.join(os.getcwd(),args.save)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
net_temp_file = os.path.join(exp_dir,'net_temp.txt')


opt['net_file'] = net_file
opt['net_temp_file'] = net_temp_file


#net_file = exp_dir+ '/net_temp.txt'

vocab_node = loader.Vocab(net_file, [0, 1])
vocab_label = loader.Vocab(label_file, [1])
vocab_feature = loader.Vocab(feature_file, [1])

opt['num_node'] = len(vocab_node)
opt['num_feature'] = len(vocab_feature)
opt['num_class'] = len(vocab_label)


graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
#import pdb; pdb.set_trace()
graph.to_symmetric(opt['self_link_weight'])
feature.to_one_hot(binary=True)
adj = graph.get_sparse_adjacency(opt['cuda'])

with open(train_file, 'r') as fi:
    idx_train = [vocab_node.stoi[line.strip()] for line in fi]
with open(dev_file, 'r') as fi:
    idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
with open(test_file, 'r') as fi:
    idx_test = [vocab_node.stoi[line.strip()] for line in fi]
idx_all = list(range(opt['num_node']))

#import pdb; pdb.set_trace()
idx_unlabeled = list(set(idx_all)-set(idx_train))
#idx_unlabeled = random.sample(idx_unlabeled, len(idx_train))
inputs = torch.Tensor(feature.one_hot)
target = torch.LongTensor(label.itol)
idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)
idx_unlabeled = torch.LongTensor(idx_unlabeled)
inputs_q = torch.zeros(opt['num_node'], opt['num_feature'])
target_q = torch.zeros(opt['num_node'], opt['num_class'])
inputs_p = torch.zeros(opt['num_node'], opt['num_class'])
target_p = torch.zeros(opt['num_node'], opt['num_class'])

if opt['cuda']:
    inputs = inputs.cuda()
    target = target.cuda()
    idx_train = idx_train.cuda()
    idx_dev = idx_dev.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    idx_unlabeled = idx_unlabeled.cuda()
    inputs_q = inputs_q.cuda()
    target_q = target_q.cuda()
    inputs_p = inputs_p.cuda()
    target_p = target_p.cuda()

gnnq = GNNq(opt, adj)
#gnnq = MLP(opt)
#gnnq = GNN_mix(opt, adj)
trainer_q = Trainer(opt, gnnq)

# Build the ema model
gnnq_ema = GNNq(opt, adj)

for ema_param, param in zip(gnnq_ema.parameters(), gnnq.parameters()):
            ema_param.data= param.data

for param in gnnq_ema.parameters():
            param.detach_()
trainer_q_ema = Trainer(opt, gnnq_ema, ema = False)




gnnp = GNNp(opt, adj)
trainer_p = Trainer(opt, gnnp)

def init_q_data():
    inputs_q.copy_(inputs)
    temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
    temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
    target_q[idx_train] = temp

def update_p_data():
    preds = trainer_q.predict(inputs_q, opt['tau'])
    if opt['draw'] == 'exp':
        inputs_p.copy_(preds)
        target_p.copy_(preds)
    elif opt['draw'] == 'max':
        idx_lb = torch.max(preds, dim=-1)[1]
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    elif opt['draw'] == 'smp':
        idx_lb = torch.multinomial(preds, 1).squeeze(1)
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        inputs_p[idx_train] = temp
        target_p[idx_train] = temp

def update_q_data():
    preds = trainer_p.predict(inputs_p)
    target_q.copy_(preds)
    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        target_q[idx_train] = temp



def update_ema_variables(model, ema_model, alpha, epoch):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (epoch + 1), alpha)
    #print (alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(final_consistency_weight, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - args.consistency_rampup_starts
    #epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight *sigmoid_rampup(epoch, args.consistency_rampup_ends - args.consistency_rampup_starts )



def sharpen(prob, temperature):
    temp_reciprocal = 1.0/ temperature
    prob = torch.pow(prob, temp_reciprocal)
    row_sum = prob.sum(dim=1).reshape(-1,1)
    out = prob/row_sum
    return out 




def pre_train(epoches):
    best = 0.0
    init_q_data()
    results = []
    
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss # remember to divide by the batch size
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss 
    
    for epoch in range(epoches):
        #loss = trainer_q.update_soft_mlp(inputs_q, target_q, idx_train)
        #loss = trainer_q.update_soft(inputs_q, target_q, idx_train)
        #import pdb; pdb.set_trace()
        ### create mix of feature and labels
        rand_index = random.randint(0,1)
        if rand_index == 0: ## do the augmented node training
            
            ## get the psudolabels for the unlabeled nodes ##
            #import pdb; pdb.set_trace()
            
            k = 10
            temp  = torch.zeros([k, target_q.shape[0], target_q.shape[1]], dtype=target_q.dtype)
            temp = temp.cuda()
            for i in range(k):
                temp[i,:,:] = trainer_q.predict_noisy(inputs_q)
            target_predict = temp.mean(dim = 0)# trainer_q.predict(inputs_q)
            
            #target_predict = trainer_q.predict(inputs_q)
            target_predict = sharpen(target_predict,0.1)
            #if epoch == 500:
            #    print (target_predict)
            target_q[idx_unlabeled] = target_predict[idx_unlabeled]
            #inputs_q_new, target_q_new, idx_train_new = get_augmented_network_input(inputs_q, target_q,idx_train,opt, net_file, net_temp_file) ## get the augmented nodes in the input space
            #idx_train_new = 
            #loss = trainer_q.update_soft_mix(inputs_q, target_q, idx_train)## for mixing features
            temp = torch.randint(0, idx_unlabeled.shape[0], size=(idx_train.shape[0],))## index of the samples chosen from idx_unlabeled
            idx_unlabeled_subset = idx_unlabeled[temp]
            loss , loss_usup= trainer_q.update_soft_aux(inputs_q, target_q, target, idx_train, idx_unlabeled_subset, adj,  opt, mixup_layer =[1])## for augmented nodes
            mixup_consistency = get_current_consistency_weight(opt['mixup_consistency'], epoch)
            total_loss = loss + mixup_consistency*loss_usup
            trainer_q.model.train()
            trainer_q.optimizer.zero_grad()
            total_loss.backward()
            trainer_q.optimizer.step()

        else:
            
            loss = trainer_q.update_soft(inputs_q, target_q, idx_train)
            
            """
            k = 10
            temp  = torch.zeros([k, target_q.shape[0], target_q.shape[1]], dtype=target_q.dtype)
            temp = temp.cuda()
            for i in range(k):
                temp[i,:,:] = trainer_q.predict_noisy(inputs_q)
            target_predict = temp.mean(dim = 0)# trainer_q.predict(inputs_q)
            target_predict = sharpen(target_predict,0.1)
            target_q[idx_unlabeled] = target_predict[idx_unlabeled]
            
            temp = torch.randint(0, idx_unlabeled.shape[0], size=(idx_train.shape[0],))## index of the samples chosen from idx_unlabeled
            idx_unlabeled_subset = idx_unlabeled[temp]
            
            loss_usup = trainer_q.update_soft(inputs_q, target_q, idx_unlabeled_subset)

            mixup_consistency = get_current_consistency_weight(opt['mixup_consistency'], epoch)
            total_loss = loss + mixup_consistency*loss_usup
            """
            
            total_loss = loss
            trainer_q.model.train()
            trainer_q.optimizer.zero_grad()
            total_loss.backward()
            trainer_q.optimizer.step()
        #loss = trainer_q.update_soft_aux(inputs_q, target_q, idx_train)## for training aux networks
        #loss_aux = loss
        #loss, loss_aux = trainer_q.update_soft_aux(inputs_q, target_q, idx_train, epoch, opt)## for auxiliary net with shared parameters

        #trainer_q.model.adj = adj
        #trainer_q.model.m1.adj = adj
        #trainer_q.model.m2.adj = adj
        _, preds, accuracy_train = trainer_q.evaluate(inputs_q, target, idx_train) ## target_new : for augmented nodes
        _, preds, accuracy_dev = trainer_q.evaluate(inputs_q, target, idx_dev)
        _, preds, accuracy_test = trainer_q.evaluate(inputs_q, target, idx_test)
        _, preds, accuracy_test_ema = trainer_q_ema.evaluate(inputs_q, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
        
        if epoch%400 == 0:
            if rand_index == 0:
                print ('epoch :{:4d},loss:{:.10f},loss_usup:{:.10f}, train_acc:{:.3f}, dev_acc:{:.3f}, test_acc:{:.3f}'.format(epoch, loss.item(),loss_usup.item(), accuracy_train, accuracy_dev, accuracy_test))
            else : 
                 print ('epoch :{:4d},loss:{:.10f}, train_acc:{:.3f}, dev_acc:{:.3f}, test_acc:{:.3f}'.format(epoch, loss.item(), accuracy_train, accuracy_dev, accuracy_test))
        
        if accuracy_dev > best:
            best = accuracy_dev
            state = dict([('model', copy.deepcopy(trainer_q.model.state_dict())), ('optim', copy.deepcopy(trainer_q.optimizer.state_dict()))])
    #trainer_q.model.load_state_dict(state['model'])
    #trainer_q.optimizer.load_state_dict(state['optim'])
        
        update_ema_variables(gnnq, gnnq_ema, opt['ema_decay'], epoch)
    
        
    return results


base_results, q_results, p_results = [], [], []
base_results += pre_train(opt['pre_epoch'])


def get_accuracy(results):
    best_dev, acc_test = 0.0, 0.0
    for d, t in results:
        if d >= best_dev:
            best_dev, acc_test = d, t
    return acc_test

acc_test = get_accuracy(base_results)

print('Test acc{:.3f}'.format(acc_test * 100))

