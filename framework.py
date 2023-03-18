import os
import torch
import torch.nn as nn
from torch import autograd
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, adjusted_rand_score
import net
import utils
from torch_scatter import scatter_mean
from utils import fix_nn


class SketchModel:
    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.dataset, opt.class_name, opt.timestamp)
        self.pretrain_dir = os.path.join(opt.checkpoints_dir, opt.dataset, opt.class_name, opt.net_name)
        self.optimizer = None
        self.loss_func = None
        self.loss = None
        self.confusion = None # confusion matrix
        self.multi_confusion = None

        self.net_name = opt.net_name
        self.net = net.init_net(opt)
        self.net.train(self.is_train)
        
        self.loss_func = torch.nn.NLLLoss().to(self.device)
        self.loss_diff = torch.nn.L1Loss().to(self.device)

        if self.is_train:
            self.opt_theta = torch.optim.Adam(self.net.module.backbone.parameters(), 
                                              lr=opt.lr, 
                                              betas=(opt.beta1, 0.999),
                                              weight_decay=opt.weight_decay)
            self.opt_phi = torch.optim.Adam(self.net.module.classifier.parameters(), 
                                              lr=opt.lr, 
                                              betas=(opt.beta1, 0.999),
                                              weight_decay=opt.weight_decay)
            self.opt_pi = torch.optim.Adam(self.net.module.creativity.parameters(), 
                                              lr=opt.lr, 
                                              betas=(opt.beta1, 0.999),
                                              weight_decay=opt.pi_weight_decay)
            self.scheduler_theta = utils.get_scheduler(self.opt_theta, opt)
            self.scheduler_phi = utils.get_scheduler(self.opt_phi, opt)
            self.scheduler_pi = utils.get_scheduler(self.opt_pi, opt)
        
        if not self.is_train: #or opt.continue_train:
            self.load_network(opt.which_epoch, mode='test')
            
        if self.is_train and opt.pretrain != '-':
            self.load_network(opt.which_epoch, mode='pretrain')
    
    def forward(self, x, edge_index, data):
        out = self.net(x, edge_index, data)
        return out

    # def backward(self, out, label):
    #     """
    #     out: (B*N, C)
    #     label: (B*N, )
    #     """
    #     self.loss = self.loss_func(out, label)
    #     self.loss.backward()

    def step_pi(self, data, data_2, data_3):
        # Preparing
        tmp_net_o = net.init_net(self.opt, is_Creativity=False)
        tmp_net_nc = net.init_net(self.opt, is_Creativity=False)
        tmp_net_o.module.backbone.load_state_dict(self.net.module.backbone.state_dict())
        tmp_net_o.module.classifier.load_state_dict(self.net.module.classifier.state_dict())
        tmp_net_nc.module.backbone.load_state_dict(self.net.module.backbone.state_dict())
        tmp_net_nc.module.classifier.load_state_dict(self.net.module.classifier.state_dict())
        tmp_opt_net_o = torch.optim.Adam(tmp_net_o.module.backbone.parameters(), 
                                            lr=self.opt.lr, 
                                            betas=(self.opt.beta1, 0.999),
                                            weight_decay=self.opt.weight_decay)
        tmp_opt_net_nc = torch.optim.Adam(tmp_net_nc.module.backbone.parameters(),
                                            lr=self.opt.lr, 
                                            betas=(self.opt.beta1, 0.999),
                                            weight_decay=self.opt.weight_decay)
        for p in tmp_net_nc.module.classifier.parameters():
            p.requires_grad = False
        for p in tmp_net_o.module.classifier.parameters():
            p.requires_grad = False

        # Step 1
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pool_edge_index'] = data.pool_edge_index.to(self.device)
        stroke_data['pos'] = x

        tmp_opt_net_nc.zero_grad()
        tmp_opt_net_o.zero_grad()

        out_nc, Feat = tmp_net_nc.module.backbone(x, edge_index, stroke_data)
        out_nc = tmp_net_nc.module.classifier(out_nc)
        offset_pred = self.net.module.creativity(Feat)
        out_o = tmp_net_o(x, edge_index, stroke_data)

        loss_net_o_main = self.loss_func(out_o, label)
        loss_net_nc_main = self.loss_func(out_nc, label) + self.opt.creativity_alpha * nn.functional.softplus(offset_pred).mean()
        grad_theta = autograd.grad(loss_net_nc_main, tmp_net_nc.module.backbone.parameters(), retain_graph=True, create_graph=True)
        fix_nn(tmp_net_nc.module.backbone, grad_theta, self.opt.lr)
        loss_net_o_main.backward()
        tmp_opt_net_o.step()

        with torch.no_grad():
            stroke_data_2= {}
            x_2 = data_2.x.to(self.device)
            label_2 = data_2.y.to(self.device)
            edge_index_2 = data_2.edge_index.to(self.device)
            stroke_data_2['stroke_idx'] = data_2.stroke_idx.to(self.device)
            stroke_data_2['batch'] = data_2.batch.to(self.device)
            stroke_data_2['edge_attr'] = data_2.edge_attr.to(self.device)
            stroke_data_2['pool_edge_index'] = data_2.pool_edge_index.to(self.device)
            stroke_data_2['pos'] = x_2

            _, Feat_nc = tmp_net_nc.module.backbone(x_2, edge_index_2, stroke_data_2)
            _, Feat_o = tmp_net_o.module.backbone(x_2, edge_index_2, stroke_data_2)


        # Step 2
        stroke_data_3= {}
        x_3 = data_3.x.to(self.device)
        label_3 = data_3.y.to(self.device)
        edge_index_3 = data_3.edge_index.to(self.device)
        stroke_data_3['stroke_idx'] = data_3.stroke_idx.to(self.device)
        stroke_data_3['batch'] = data_3.batch.to(self.device)
        stroke_data_3['edge_attr'] = data_3.edge_attr.to(self.device)
        stroke_data_3['pool_edge_index'] = data_3.pool_edge_index.to(self.device)
        stroke_data_3['pos'] = x_3

        new_out_nc = tmp_net_nc(x_3, edge_index_3, stroke_data_3)
        offset_nc = self.loss_func(new_out_nc, label_3)
        with torch.no_grad():
            new_out_o = tmp_net_o(x_3, edge_index_3, stroke_data_3)
            offset_o = self.loss_func(new_out_o, label_3)
        loss_creativity = self.opt.creativity_beta * (self.loss_diff((self.net.module.creativity(Feat_o.detach()) - self.net.module.creativity(Feat_nc.detach())).mean(), (offset_o - offset_nc).detach())
                        + nn.functional.softplus(offset_nc - offset_o.detach()))
        print("offset:", (offset_o - offset_nc).mean().data)
        print("offset_o_pred:", self.net.module.creativity(Feat_o.detach()).mean().data)
        print("offset_nc_pred:", self.net.module.creativity(Feat_nc.detach()).mean().data)
        print("loss_creativity:", loss_creativity.data.mean())
        self.opt_pi.zero_grad()
        loss_creativity.backward()
        self.opt_pi.step()
        # torch.cuda.empty_cache()
    

    def step_theta(self, data):
        """
        """
        # Step 3
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pool_edge_index'] = data.pool_edge_index.to(self.device)
        stroke_data['pos'] = x

        self.opt_theta.zero_grad()
        self.opt_phi.zero_grad()
        out, c_score = self.forward(x, edge_index, stroke_data)
        self.loss = self.loss_func(out, label) + self.opt.creativity_alpha * nn.functional.softplus(c_score).mean()
        self.loss.backward()
        self.opt_theta.step()
        self.opt_phi.step()
        # torch.cuda.empty_cache()

    def test_time(self, data):
        """
        x: (B*N, F)
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pool_edge_index'] = data.pool_edge_index.to(self.device)
        stroke_data['pos'] = x

        with torch.no_grad():
            out = self.forward(x, edge_index, stroke_data)
        
        return out
    
    def test(self, data, if_eval=False):
        """
        x: (B*N, F)
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pool_edge_index'] = data.pool_edge_index.to(self.device)
        stroke_data['pos'] = x

        with torch.no_grad():
            out = self.forward(x, edge_index, stroke_data)
            predict = torch.argmax(out, dim=1).cpu().numpy()

            if (label < 0).any(): # for meaningless label
                self.loss = torch.Tensor([0])
            else:
                self.loss = self.loss_func(out, label)

        return self.loss, predict
        
    
    def print_detail(self):
        print(self.net)

    def update_learning_rate(self):
        """
        update learning rate (called once every epoch)
        """
        self.scheduler_theta.step()
        self.scheduler_phi.step()
        self.scheduler_pi.step()
        lr = self.opt_theta.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_network(self, epoch):
        """
        save model to disk
        """
        path = os.path.join(self.save_dir, 
                            '{}_{}.pkl'.format(self.net_name, epoch))
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), path)
    
    def load_network(self, epoch, mode='test'):
        """
        load model from disk
        """
        path = os.path.join(self.save_dir if mode =='test' else self.pretrain_dir, 
                            '{}_{}.pkl'.format(self.net_name, epoch))
        
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from {}'.format(path))
        state_dict = torch.load(path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


