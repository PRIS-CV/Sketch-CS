import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.conv import *
from blocks.basic import MLPLinear, MultiSeq
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_cluster import knn_graph
# import torchsnooper
import torch_geometric.nn as tgnn


def init_net(opt, is_Creativity=True):
    if opt.net_name == 'SketchGNN_Feature':
        net = SketchGNN_Feature(opt)
    elif opt.net_name == 'SketchGNN_Classifier':
        net = SketchGNN_Classifier(opt)
    elif opt.net_name == 'SketchGNN_CreativitySeg':
        net = SketchGNN_CreativitySeg(opt, is_Creativity)
    else:
        raise NotImplementedError('net {} is not implemented. Please check.\n'.format(opt.net_name))
    
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(opt.gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=opt.gpu_ids)
    return net


class SketchGNN_Feature(nn.Module):
    def __init__(self, opt):
        super(SketchGNN_Feature, self).__init__()
        self.opt = opt
        self.n_blocks = opt.n_blocks
        self.channels = opt.channels
        self.pool_channels = opt.pool_channels

        ####################### point feature #######################
        opt.kernel_size = opt.local_k
        opt.dilation = opt.local_dilation
        opt.stochastic = opt.local_stochastic
        opt.epsilon = opt.local_epsilon
        dilations = [1, 4, 8] + [opt.local_dilation] * (self.n_blocks-2)   
        
        # head
        if self.opt.local_adj_type == 'static':
            self.local_head = GraphConv(opt.in_feature, self.channels, opt)
        else:
            self.local_head = DynConv(opt.in_feature, self.channels, dilations[0], opt) 
        
        # local backbone
        self.local_backbone = MultiSeq(*[ResGcnBlock(self.channels, opt.local_adj_type, dilations[i+1], opt) for i in range(self.n_blocks)])

        ####################### stroke & sketch feature #######################
        opt.kernel_size = opt.global_k
        opt.dilation = opt.global_dilation
        opt.stochastic = opt.global_stochastic
        opt.epsilon = opt.global_epsilon
        dilations = [1, opt.global_dilation//4, opt.global_dilation//2] + [opt.global_dilation] * (self.n_blocks-2)   
        
        # head
        if self.opt.global_adj_type == 'static':
            self.global_head = GraphConv(opt.in_feature, self.channels, opt)
        else:
            self.global_head = DynConv(opt.in_feature, self.channels, dilations[0], opt)    
        
        # global backbone
        self.global_backbone = MultiSeq(*[ResGcnBlock(self.channels, opt.global_adj_type, dilations[i+1], opt) for i in range(self.n_blocks)])
        
        if opt.fusion_type == 'mix':
            self.pool = MixPool(opt.channels*(opt.n_blocks+1), opt.pool_channels // 2)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        elif opt.fusion_type == 'max':
            self.pool = MaxPool(opt.channels*(opt.n_blocks+1), opt.pool_channels)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        else:
            raise NotImplementedError('fusion_type {} is not implemented. Please check.\n'.format(opt.fusion_type))
        self.fuse_segment = MLPLinear(mlpSegment[:2], norm_type='batch', act_type='relu')
        
    # @torchsnooper.snoop()
    def forward(self, x, edge_index, data):
        """
        x: (BxN) x F
        """
        BN = x.shape[0]
        ####################### local line #######################
        x_l = self.local_head(x, edge_index, data).unsqueeze(-1)
        x_l = torch.cat((x_l, x_l), 2)
        x_l = self.local_backbone(x_l, edge_index, data)[0][:,:,1:].contiguous().view(BN, -1)

        ####################### global line #######################
        x_g = self.global_head(x, edge_index, data).unsqueeze(-1)
        x_g = torch.cat((x_g, x_g), 2)
        x_g = self.global_backbone(x_g, edge_index, data)[0][:,:,1:].contiguous().view(BN, -1)
        x_g = self.pool(x_g, data['stroke_idx'], data['batch'])

        ####################### segment #######################
        x = torch.cat([x_l, x_g], dim=1)
        # Feat = torch.cat((tgnn.global_max_pool(x, data['batch']), tgnn.global_mean_pool(x, data['batch'])), dim=1)
        x = self.fuse_segment(x)
        Feat = torch.cat((tgnn.global_max_pool(x, data['batch']), tgnn.global_mean_pool(x, data['batch'])), dim=1)
        return x, Feat.unsqueeze(-1)


class SketchGNN_Classifier(nn.Module):
    def __init__(self, opt):
        super(SketchGNN_Classifier, self).__init__()
        self.opt = opt
        self.n_blocks = opt.n_blocks
        self.channels = opt.channels
        self.pool_channels = opt.pool_channels

        ####################### point feature #######################
        opt.kernel_size = opt.local_k
        opt.dilation = opt.local_dilation
        opt.stochastic = opt.local_stochastic
        opt.epsilon = opt.local_epsilon
        dilations = [1, 4, 8] + [opt.local_dilation] * (self.n_blocks-2)   
        
        # head
        # if self.opt.local_adj_type == 'static':
        #     self.local_head = GraphConv(opt.in_feature, self.channels, opt)
        # else:
        #     self.local_head = DynConv(opt.in_feature, self.channels, dilations[0], opt) 
        
        # # local backbone
        # self.local_backbone = MultiSeq(*[ResGcnBlock(self.channels, opt.local_adj_type, dilations[i+1], opt) for i in range(self.n_blocks)])

        ####################### stroke & sketch feature #######################
        opt.kernel_size = opt.global_k
        opt.dilation = opt.global_dilation
        opt.stochastic = opt.global_stochastic
        opt.epsilon = opt.global_epsilon
        dilations = [1, opt.global_dilation//4, opt.global_dilation//2] + [opt.global_dilation] * (self.n_blocks-2)   
        
        # head
        # if self.opt.global_adj_type == 'static':
        #     self.global_head = GraphConv(opt.in_feature, self.channels, opt)
        # else:
        #     self.global_head = DynConv(opt.in_feature, self.channels, dilations[0], opt)    
        
        # # global backbone
        # self.global_backbone = MultiSeq(*[ResGcnBlock(self.channels, opt.global_adj_type, dilations[i+1], opt) for i in range(self.n_blocks)])
        
        if opt.fusion_type == 'mix':
            # self.pool = MixPool(opt.channels*(opt.n_blocks+1), opt.pool_channels // 2)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        elif opt.fusion_type == 'max':
            # self.pool = MaxPool(opt.channels*(opt.n_blocks+1), opt.pool_channels)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        else:
            raise NotImplementedError('fusion_type {} is not implemented. Please check.\n'.format(opt.fusion_type))
        self.segment = MultiSeq(*[MLPLinear(mlpSegment[1:], norm_type='batch', act_type='relu'),
                                  MLPLinear([mlpSegment[-1], opt.out_segment], norm_type='batch', act_type=None)])
        # softmax        
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        
    # @torchsnooper.snoop()
    def forward(self, x):
        """
        x: (BxN) x F
        """
        x = self.segment(x)
        return self.LogSoftmax(x)


class SketchGNN_CreativitySeg(nn.Module):
    def __init__(self, opt, is_Creativity=True):
        super(SketchGNN_CreativitySeg, self).__init__()
        self.points_num = opt.points_num
        self.n_blocks = opt.n_blocks
        self.channels = opt.channels
        self.in_feature = opt.in_feature
        self.out_segment = opt.out_segment
        self.pool_channels = opt.pool_channels
        self.is_Creativity = is_Creativity
        self.backbone = SketchGNN_Feature(opt)
        self.classifier = SketchGNN_Classifier(opt)
        if self.is_Creativity:
            self.creativity = nn.Sequential(nn.Conv1d(opt.mlp_segment[0]*2, opt.mlp_segment[0]*2//4, kernel_size=1),
                                        nn.BatchNorm1d(opt.mlp_segment[0]*2//4),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(opt.mlp_segment[0]*2//4, 1, kernel_size=1),
                                        # nn.Sigmoid()
                                    )

    def forward(self, x, edge_index, data):
        if self.is_Creativity:
            if self.training:
                x, Feat = self.backbone(x, edge_index, data)
                c_score = self.creativity(Feat)
                x = self.classifier(x)
                return x, c_score.view(-1)
            else:
                x, _ = self.backbone(x, edge_index, data)
                x = self.classifier(x)
                return x
        else:
            x, _ = self.backbone(x, edge_index, data)
            x = self.classifier(x)
            return x

