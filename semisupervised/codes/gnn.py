import math
import numpy as np
import random
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import os
from layer import GraphConvolution
import loader
from shutil import copyfile

def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    #y_a, y_b = y, y[index]
    mixed_y = lam * y + (1 - lam) * y[index,:]
    return mixed_x, mixed_y

def mixup_gnn_hidden(x, target, train_idx, alpha):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)#, size = train_idx.shape[0])#+0.001
    else:
        lam = 1.
        #lam = np.repeat(lam, train_idx.shape[0])
    #import pdb; pdb.set_trace()
    #lam = lam.reshape(train_idx.shape[0],1)
    #lam = torch.from_numpy(lam).cuda().type(torch.float32)
    permuted_train_idx = train_idx[torch.randperm(train_idx.shape[0])]
    x[train_idx] = lam*x[train_idx]+ (1-lam)*x[permuted_train_idx]
    #target[train_idx] = lam*target[train_idx]+ (1-lam)*target[permuted_train_idx]
    """
    ### mix the unlabeded nodes###
    all_idx = set(range(0,x.shape[0]))
    train_idx_set = set(list(train_idx.cpu().numpy()))
    unlabeled_idx = np.asarray(list(all_idx-train_idx_set))
    permuted_unlabeled_idx = unlabeled_idx[torch.randperm(unlabeled_idx.shape[0])]
    x[unlabeled_idx] = lam*x[unlabeled_idx]+ (1-lam)*x[permuted_unlabeled_idx]
    """
    return x, target[train_idx], target[permuted_train_idx],lam

class GNN_mix(nn.Module):
    def __init__(self, opt, adj):
        super(GNN_mix, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', 1000)])
        self.m1 = GraphConvolution(opt_, adj)

        #self.linear_m1_1 = nn.Linear(1000,500)
        #self.linear_m1_2 = nn.Linear(50,opt['num_class'] )
        
        opt_ = dict([('in', 1000), ('out', 500)])
        self.m2 = GraphConvolution(opt_, adj)
        
        #self.linear_m2_1 = nn.Linear(50,20)
        #self.linear_m2_2 = nn.Linear(20,opt['num_class'] )
        
        opt_ = dict([('in', 500), ('out', 100)])
        self.m3 = GraphConvolution(opt_, adj)
        
        #self.linear_m3_1 = nn.Linear(10,5 )
        #self.linear_m3_2 = nn.Linear(5,opt['num_class'] )

        opt_ = dict([('in', 100), ('out', opt['num_class'])])
        self.m4 = GraphConvolution(opt_, adj)
        """
        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)
        """  
        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()
    
    def forward(self, x, target=None, train_idx= None, mixup_input=False, mixup_hidden= False,  mixup_alpha = 0.0, layer_mix = None):
        """    
        #import pdb; pdb.set_trace()
        if target is not None: 
            x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        if target is not None: 
            x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        if target is not None:
            return x, target_a, target_b, lam
        else: 
            return x
        """
        #import pdb; pdb.set_trace()        
        if mixup_hidden == True or mixup_input == True:
            if mixup_hidden == True:
                layer_mix = random.randint(1,layer_mix)
            elif mixup_input == True:
                layer_mix = 0

    
            if layer_mix ==0:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)

            x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    
            x = self.m1(x)
            x = F.relu(x)
            if layer_mix == 1:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)

            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = self.m2(x)
            x = F.relu(x)

            if layer_mix == 2:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)
        
            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = self.m3(x)
            x = F.relu(x)

            if layer_mix == 3:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)

            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = self.m4(x)

            return x, target_a, target_b, lam
        else:
            x = F.dropout(x, self.opt['input_dropout'], training=self.training)
            x = self.m1(x)
            x = F.relu(x)
            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = self.m2(x)
            x = F.relu(x)
            x = F.dropout(x, self.opt['input_dropout'], training=self.training)
            x = self.m3(x)
            x = F.relu(x)
            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = self.m4(x)
            return x
    """
    def get_m1_mix(self, x,target=None, train_idx= None, mixup_alpha = 0.0):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        ## mix h1 ##
        x, target_a, target_b,lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)
        x = self.linear_m1_1(x)
        x = F.relu(x)
        x = self.linear_m1_2(x)
        return x, target_a, target_b, lam
   
   
    def get_m2_mix(self, x,target=None, train_idx= None, mixup_alpha = 0.0):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        x = F.relu(x)
        ## mix h2 ##
        x, target_a, target_b,lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)
        x = self.linear_m2_1(x)
        x = F.relu(x)
        x = self.linear_m2_2(x)
        return x, target_a, target_b, lam

    def get_m3_mix(self, x,target=None, train_idx= None, mixup_alpha = 0.0):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m3(x)
        x = F.relu(x)
        ## mix h3
        x, target_a, target_b,lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)
        x = self.linear_m3_1(x)
        x = F.relu(x)
        x = self.linear_m3_2(x)
        return x, target_a, target_b, lam
    """
    
    

def get_augmented_network_input(model, inputs_q, target_q, target, idx_train,opt):
    # idx_train : this is the index of the nodes that will be mixed
    # pass the train node idx ( labeled nodes) or the unlabeled node idx here

    ### create a new net file###
    if os.path.exists(opt['net_temp_file']):
        os.remove(opt['net_temp_file'])
        copyfile(opt['net_file'], opt['net_temp_file'])
    else:
        copyfile(opt['net_file'], opt['net_temp_file'])
    
    lamb = np.random.beta(opt['mixup_alpha'],opt['mixup_alpha'])
    #import pdb; pdb.set_trace()
    inputs_q_new = inputs_q
    target_q_new = target_q
    idx_train_new = torch.tensor([], dtype= idx_train.dtype).cuda()# idx_train# [] for not adding the original idx_train in the additional train data
    target_new = target
    
    for j in range(1):
        permuted_train_idx = idx_train[torch.randperm(idx_train.shape[0])]
        train_x_additional = lamb*inputs_q[idx_train]+ (1-lamb)*inputs_q[permuted_train_idx]
        train_y_additional = lamb*target_q[idx_train]+ (1-lamb)*target_q[permuted_train_idx]
        idx_train_additional = np.arange(idx_train.shape[0])
        idx_train_additional = torch.from_numpy(idx_train_additional)
        idx_train_additional = idx_train_additional.cuda()
        idx_train_additional = idx_train_additional + target_q_new.shape[0]

        inputs_q_new = torch.cat((inputs_q_new, train_x_additional),0)
        target_q_new = torch.cat((target_q_new, train_y_additional),0)
        idx_train_new = torch.cat((idx_train_new, idx_train_additional),0)
    
        ## add dummy labels to the target tensor, these dummy values will not be used so I just used '0'##
        #import pdb; pdb.set_trace()
        
        temp = torch.zeros(train_y_additional.shape[0], dtype = target.dtype)
        temp = temp.cuda()
        target_new = torch.cat((target_new, temp),0)

        #import pdb; pdb.set_trace()
        fi = open(opt['net_temp_file'], 'a+')
        start_index_for_additional_nodes = target_q.shape[0]+j*idx_train.shape[0]
        for i in range(idx_train.shape[0]):
            node_index = start_index_for_additional_nodes+i
            fi.write(str(node_index)+'\t'+str(idx_train[i].item())+'\t'+str(1)+'\n')
            fi.write(str(idx_train[i].item())+'\t'+str(node_index)+'\t'+str(1)+'\n')
            fi.write(str(node_index)+'\t'+str(permuted_train_idx[i].item())+'\t'+str(1)+'\n')
            fi.write(str(permuted_train_idx[i].item())+'\t'+str(node_index)+'\t'+str(1)+'\n')
        fi.close()
    

    ## reload the net file in the adjacency matrix###
    vocab_node = loader.Vocab(opt['net_temp_file'], [0, 1])
    graph = loader.Graph(file_name=opt['net_temp_file'], entity=[vocab_node, 0, 1])
    graph.to_symmetric(opt['self_link_weight'])
    adj_new = graph.get_sparse_adjacency(opt['cuda'])
    model.m1.adj = adj_new
    model.m2.adj = adj_new
    #trainer_q.model.adj = adj_new
    #trainer_q.model.m1.adj = adj_new
    #trainer_q.model.m2.adj = adj_new
    #trainer_q.model.m3.adj = adj
    #trainer_q.model.m4.adj = adj
    
    return inputs_q_new, target_q_new, idx_train_new


class GNNq(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)
        
        #opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        #self.m3 = GraphConvolution(opt_, adj)### used for auxiliary network. it will be used a fully-connected layer. for ease of implementation I used GCN layer.

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x
    
    
    def forward_mix(self, x, target, target_discrete, idx, opt, mixup_layer):
        layer = random.choice(mixup_layer)
        if layer == 0:
            x, target, idx = get_augmented_network_input(self, x, target, target_discrete, idx,opt)
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        if layer == 1:
            x, target, idx = get_augmented_network_input(self, x, target, target_discrete, idx,opt)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x, target, idx
    
    def forward_aux(self, x, target=None, train_idx= None, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None):
        
        if mixup_hidden == True or mixup_input == True:
            if mixup_hidden == True:
                layer_mix = random.choice(layer_mix)
            elif mixup_input == True:
                layer_mix = 0

    
            if layer_mix ==0:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)

            x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    
            x = self.m1.forward_aux(x)
            x = F.relu(x)
            if layer_mix == 1:
                x, target_a, target_b, lam = mixup_gnn_hidden(x, target, train_idx, mixup_alpha)

            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = self.m2.forward_aux(x)
            
            return x, target_a, target_b, lam
        
        else:
        
            x = F.dropout(x, self.opt['input_dropout'], training=self.training)
            x = self.m1.forward_aux(x)
            x = F.relu(x)
            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = self.m2.forward_aux(x)
            return x

class GNNp(nn.Module):
    def __init__(self, opt, adj):
        super(GNNp, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_class']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if  opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, opt):
        super(MLP, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(opt['num_feature'],500) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 250) 
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(250, 100)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(50, opt['hidden_dim'])
        self.relu = nn.ReLU()
        self.fc6 = nn.Linear(opt['hidden_dim'],opt['num_class'] )
        
        if opt['cuda']:
            self.cuda()
        
    
    def forward(self, x, target=None, mixup_input = False, mixup_hidden = False,  mixup_alpha = 0.1, layer_mix=None):
        
        if mixup_hidden == True:
            layer_mix = random.randint(1,layer_mix)
        elif mixup_input == True:
            layer_mix = 0
        
        out = x
        if layer_mix == 0:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc1(x)
        out = self.relu(out)
        
        if layer_mix == 1:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc2(out)
        out = self.relu(out)
        if layer_mix == 2:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc3(out)
        out = self.relu(out)
        if layer_mix == 3:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc4(out)
        out = self.relu(out)
        if layer_mix == 4:
            out, mixed_target = mixup_data(out, target, mixup_alpha)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        if layer_mix == None:
            return out
        else:
            return out, mixed_target


        
