import torch
from torch_geometric.nn.conv import GraphConv
from GTM.GTM import GTM
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadd
from torch.functional import F
from torch.nn.init import eye_
import numpy as np
import os


class GNN_Conv_GTM(torch.nn.Module):
    """
    MUTAG input = loader_train.dataset.num_features, n_units, n_classes, gtm_grids_dim, gtm_rbf, drop_prob
    """
    def __init__(self, in_channels, out_channels, n_class=2, gtm_grid_dims=(10, 10), rbf=10, dropout=0, device=None):
        super(GNN_Conv_GTM, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels # equals number of features
        self.out_channels = out_channels # equals number of neurons (30)
        self.n_class = n_class
        self.gtm_grid_dims = gtm_grid_dims

        self.conv0 = GraphConv(self.in_channels, self.in_channels, bias=False) #TODO This is never used?? why here bias (and req_grad) is False? Trick per passare da sparso onehot a valore reale più denso

        self.conv1 = GraphConv(self.in_channels, self.out_channels)
        #self.conv2 = GraphConv(self.out_channels, out_channels * 2)
        #self.conv3 = GraphConv(self.out_channels * 2, out_channels * 3)
        self.conv2 = GraphConv(self.out_channels, out_channels)
        self.conv3 = GraphConv(self.out_channels, out_channels )

        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()
        self.act3 = torch.nn.LeakyReLU()

        self.norm1 = torch.nn.BatchNorm1d(self.out_channels)
        self.norm2 = torch.nn.BatchNorm1d(self.out_channels) #* 2)
        self.norm3 = torch.nn.BatchNorm1d(self.out_channels) #* 3)

        self.dropout = torch.nn.Dropout(p=dropout)

        # define readout of GNN_only model
        #self.lin_GNN = torch.nn.Linear((out_channels + out_channels * 2 + self.out_channels * 3) * 3, n_class)
        self.lin_GNN = torch.nn.Linear(out_channels * 3 * 3, n_class)

        # define gtm for read out
        self.gtm1 = GTM(out_channels, out_size=gtm_grid_dims, m=rbf, method='full_prob', device=self.device, sigma=1)
        self.gtm2 = GTM(out_channels , out_size=gtm_grid_dims, m=rbf, method='full_prob', device=self.device, sigma=1) # outchannel * 2
        self.gtm3 = GTM(out_channels , out_size=gtm_grid_dims, m=rbf, method='full_prob', device=self.device, sigma=1) # outchannel * 3

        # define read_out # TODO fix hardcoded dimension from GTM output (coming from MatM), or set to 2 for 2D punctual result (mean)
        self.out_conv1 = GraphConv(gtm_grid_dims[0] * gtm_grid_dims[1], self.out_channels)
        self.out_conv2 = GraphConv(gtm_grid_dims[0] * gtm_grid_dims[1], self.out_channels)
        self.out_conv3 = GraphConv(gtm_grid_dims[0] * gtm_grid_dims[1], self.out_channels)

        self.out_norm1 = torch.nn.BatchNorm1d(self.out_channels)
        self.out_norm2 = torch.nn.BatchNorm1d(self.out_channels)
        self.out_norm3 = torch.nn.BatchNorm1d(self.out_channels)

        self.lin_out = torch.nn.Linear(self.out_channels * 3 * 3, n_class) # TODO output ha prima dimensione il batch e seconda il one hot per classe
        self.out_fun = torch.nn.LogSoftmax(dim=1)

        # for name, param in self.named_parameters():
        #     print('name: ', name)
        #     print(type(param))
        #     print('param.shape: ', param.shape)
        #     print('param.requires_grad: ', param.requires_grad)
        #     print('=====')

        self.reset_prameters()

    def reset_prameters(self):
        # --- previous versions ---
        # eye_(self.conv0.weight) #lin_l = weight (1.3.2)
        # self.conv0.weight.requires_grad = False
        # eye_(self.conv0.lin.weight) # lin_r = lin (1.3.2)

        eye_(self.conv0.lin_l.weight)
        self.conv0.lin_l.weight.requires_grad = False
        eye_(self.conv0.lin_r.weight)  # lin_r = lin (1.3.2)

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

        self.lin_GNN.reset_parameters()

        self.out_conv1.reset_parameters()
        self.out_conv2.reset_parameters()
        self.out_conv3.reset_parameters()

        self.out_norm1.reset_parameters()
        self.out_norm2.reset_parameters()
        self.out_norm3.reset_parameters()

        self.lin_out.reset_parameters()

    def get_gtm_weights(self):
        return self.gtm1.weight, self.gtm2.weight, self.gtm3.weight

    def forward(self, data, conv_train=False, gtm_train=False):

        x = data.x

        edge_index = data.edge_index

        x1 = self.norm1(self.act1(self.conv1(x, edge_index))) # Eq (6): We start considering graph convolution layers to provide a representation for each node in the graphs
        x = self.dropout(x1)

        x2 = self.norm2(self.act2(self.conv2(x, edge_index))) # Eq (7): We stack 3 graph convolution layers
        x = self.dropout(x2)

        x3 = self.norm3(self.act3(self.conv3(x, edge_index)))

        h_conv = torch.cat([x1, x2, x3], dim=1)

        # compute GNN only output
        conv_batch_avg = gap(h_conv, data.batch)
        conv_batch_add = gadd(h_conv, data.batch)
        conv_batch_max = gmp(h_conv, data.batch)

        h_GNN = torch.cat([conv_batch_avg, conv_batch_add, conv_batch_max], dim=1) # Eq (22)

        gnn_out = self.out_fun(self.lin_GNN(h_GNN)) # Eq (23,24)

        if conv_train:
            return None, None, gnn_out

        # keep only gnn_out & h_conv
        del h_GNN, conv_batch_avg, conv_batch_add, conv_batch_max # TODO is it logically correct? wtb backpropagation?

        # GTM
        _, gtm_out_1 = self.gtm1(x1) # Eq (8)
        _, gtm_out_2 = self.gtm2(x2) # gtm_out has size (#node representations ,#latent variables)
        _, gtm_out_3 = self.gtm3(x3)

        if gtm_train:
            return None, h_conv, None

        with torch.no_grad(): # Set Max = 1 for GTM outputs (so they are bounded in [0,1])
            gtm_out_1 /= gtm_out_1.max(axis=1, keepdim=True).values
            gtm_out_2 /= gtm_out_2.max(axis=1, keepdim=True).values
            gtm_out_3 /= gtm_out_3.max(axis=1, keepdim=True).values


        # READOUT
        h1 = self.out_norm1(self.act1(self.out_conv1(gtm_out_1, edge_index))) # Eq (9)
        h2 = self.out_norm2(self.act2(self.out_conv2(gtm_out_2, edge_index)))
        h3 = self.out_norm3(self.act3(self.out_conv3(gtm_out_3, edge_index)))

        gtm_out_conv = torch.cat([h1, h2, h3], dim=1)

        gtm_batch_avg = gap(gtm_out_conv, data.batch) # Aggregates w.r.t belonging batch
        gtm_batch_add = gadd(gtm_out_conv, data.batch)
        gtm_batch_max = gmp(gtm_out_conv, data.batch)

        h = torch.cat([gtm_batch_avg, gtm_batch_add, gtm_batch_max], dim=1) # Eq (11)

        h = self.out_fun(self.lin_out(h))

        return h, h_conv, gnn_out
