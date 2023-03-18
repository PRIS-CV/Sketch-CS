import os
import numpy as np
from dataset import SketchDataset
import torch
from torch_geometric.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def fix_nn(model, theta, lr):
    num_grad = 0
    def k_param_fn(tmp_model, lr, name=None):
        nonlocal num_grad
        if len(tmp_model._modules)!=0:
            for(k,v) in tmp_model._modules.items():
                if name is None:
                    k_param_fn(v, lr, name=str(k))
                else:
                    k_param_fn(v, lr, name=str(name+'.'+k))
        else:
            for (k,v) in tmp_model._parameters.items():
                if not isinstance(v,torch.Tensor):
                    continue
                if theta[num_grad] is None:
                    num_grad+=1
                    continue
                else:
                    tmp_model._parameters[k] = tmp_model._parameters[k].data - lr * theta[num_grad]
                    num_grad+=1

    k_param_fn(model, lr)
    if num_grad != len(theta):
        print("Warning!!!Theta Length.")
        import pdb; pdb.set_trace()
    return model


def load_data(opt, datasetType='train', shuffle=False):
    data_set = SketchDataset(
        opt=opt,
        root=os.path.join('./dataset', opt.dataset),
        class_name=opt.class_name,
        split=datasetType
    )
    data_loader = DataLoader(
        data_set,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=opt.num_workers
    )
    return data_loader


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch, eta_min=opt.eta_min)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def build_record(recode, opt):
    # recode = {}
    # recode['timestamp'] = opt.timestamp
    # recode['class_name'] = opt.class_name
    # recode['net_name'] = opt.net_name
    recode['dataset'] = opt.dataset
    recode['in_feature'] = opt.in_feature
    net_structure = {}
    net_structure['n_blocks'] = opt.n_blocks
    net_structure['channels'] = opt.channels
    net_structure['gcn_type'] = opt.gcn_type
    net_structure['mixedge'] = opt.mixedge
     
    if opt.net_name == 'SketchGNN_CreativitySeg': # SketchTwoLine
        net_structure['local_adj_type'] = opt.local_adj_type
        net_structure['global_adj_type'] = opt.global_adj_type
        if opt.local_adj_type == 'dynamic':
            net_structure['local_k'] = opt.local_k
            net_structure['local_dilation'] = opt.local_dilation
            net_structure['local_stochastic'] = opt.local_stochastic
            net_structure['local_epsilon'] = opt.local_epsilon
        if opt.global_adj_type == 'dynamic':
            net_structure['global_k'] = opt.global_k
            net_structure['global_dilation'] = opt.global_dilation
            net_structure['global_stochastic'] = opt.global_stochastic
            net_structure['global_epsilon'] = opt.global_epsilon
    else:
        raise NotImplementedError('net  {} is not implemented. Please check.\n'.format(opt.net_name))
    
    net_structure['fusion_type'] = opt.fusion_type
    net_structure['block_type'] = opt.block_type
    net_structure['mlp_segment'] = opt.mlp_segment
    net_structure['pool_channels'] = opt.pool_channels
    net_structure['act_type'] = opt.act_type
    net_structure['norm_type'] = opt.norm_type
    net_structure['alpha'] = opt.alpha
    recode['net_structure'] = net_structure

    train_message = {}
    train_message['creativity-alpha'] = opt.creativity_alpha
    train_message['creativity-beta'] = opt.creativity_beta
    train_message['epoch'] = opt.epoch
    train_message['batch_size'] = opt.batch_size
    train_message['lr'] = opt.lr
    train_message['lr_policy'] = opt.lr_policy
    train_message['lr_decay_iters'] = opt.lr_decay_iters
    train_message['beta1'] = opt.beta1
    if opt.disorder_id:
        train_message['disorder_id'] = opt.disorder_id
    if opt.shuffle:
        train_message['shuffle'] = opt.shuffle
        train_message['seed'] = opt.seed
    train_message['dataset'] = opt.dataset
    train_message['train_dataset'] = opt.train_dataset
    train_message['valid_dataset'] = opt.valid_dataset
    train_message['pretrain'] = opt.pretrain
    train_message['trainlist'] = opt.trainlist
    train_message['train_log_name'] = opt.train_log_name
    train_message['random_iter'] = opt.random_iter
    recode['train_message'] = train_message
    recode['eval_way'] = opt.eval_way
    recode['metric_way'] = opt.metric_way
    recode['comment'] = opt.comment
    return recode
