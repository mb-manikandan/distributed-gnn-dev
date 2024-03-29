from typing import Callable
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import SAGEConv, GATConv, GINConv
from torch_geometric.nn import JumpingKnowledge, GCNConv, ARMAConv

# from fast_trainer.transferers import DeviceIterator


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        # m.bias.data.fill_(0.01)


# Point2: Define DNN Models

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        kwargs = dict(bias=False)
        conv_layer = SAGEConv
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(
                hidden_channels, hidden_channels, **kwargs))
        self.convs.append(conv_layer(
            hidden_channels, hidden_channels, **kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            conv.apply(init_weights)

    def forward(self, x, adjs):
        x = x.to(torch.float)
        end_size = adjs[-1][-1][1]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return torch.log_softmax(x, dim=-1)


class SAGEClassic(torch.nn.Module):
    conv_layer = SAGEConv

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.conv_layer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(self.conv_layer(hidden_channels, hidden_channels))
        self.convs.append(self.conv_layer(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        x = x.to(torch.float)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)


# Needed by SAGEResInception
class MLP(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 embed_dim,
                 num_layers: int,
                 act: str = 'ReLU',
                 bn: bool = False,
                 end_up_with_fc=False,
                 bias=True):
        super(MLP, self).__init__()
        self.module_list = []

        for i in range(num_layers):
            d_in = input_dim if i == 0 else hidden_dim
            d_out = embed_dim if i == num_layers - 1 else hidden_dim
            self.module_list.append(torch.nn.Linear(d_in, d_out, bias=bias))
            if end_up_with_fc:
                continue
            if bn:
                self.module_list.append(torch.nn.BatchNorm1d(d_out))
            self.module_list.append(getattr(torch.nn, act)(True))
        self.module_list = torch.nn.Sequential(*self.module_list)

    def reset_parameters(self):
        for x in self.module_list:
            if hasattr(x, "reset_parameters"):
                x.reset_parameters()

    def forward(self, x):
        return self.module_list(x)


class SAGEResInception(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        conv_layer = SAGEConv
        kwargs = dict(bias=False)
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.res_linears = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.res_linears.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(
                hidden_channels, hidden_channels, **kwargs))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.res_linears.append(torch.nn.Identity())
        self.convs.append(conv_layer(
            hidden_channels, hidden_channels, **kwargs))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.res_linears.append(torch.nn.Identity())

        self.mlp = MLP(in_channels + hidden_channels * (num_layers),
                       2*out_channels, out_channels,
                       num_layers=2, bn=True, end_up_with_fc=True,
                       act='LeakyReLU')
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            conv.apply(init_weights)
        for x in self.res_linears:
            if isinstance(x, torch.nn.Linear):
                x.reset_parameters()
        for x in self.bns:
            x.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, _x, adjs):
        _x = _x.to(torch.float)
        collect = []
        end_size = adjs[-1][-1][1]
        x = F.dropout(_x, p=0.1, training=self.training)
        collect.append(x[:end_size])
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((F.dropout(x, p=0.1, training=self.training),
                               F.dropout(x_target, p=0.1,
                                         training=self.training)), edge_index)
            x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            collect.append(x[:end_size])
            x += self.res_linears[i](x_target)
        return torch.log_softmax(self.mlp(torch.cat(collect, -1)), dim=-1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        kwargs = dict(bias=False, heads=1)
        conv_layer = GATConv
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(
                hidden_channels, hidden_channels, **kwargs))
        self.convs.append(conv_layer(hidden_channels, out_channels, **kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            conv.apply(init_weights)

    def forward(self, x, adjs):
        x = x.to(torch.float)
        end_size = adjs[-1][-1][1]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return torch.log_softmax(x, dim=-1)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        kwargs = dict()
        conv_layer = GINConv
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.convs.append(GINConv(Sequential(
            Linear(in_channels, hidden_channels),
            BatchNorm1d(hidden_channels), ReLU(),
            Linear(hidden_channels, hidden_channels), ReLU())))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels), ReLU(),
                Linear(hidden_channels, hidden_channels), ReLU())))
        self.convs.append(GINConv(Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels), ReLU(),
            Linear(hidden_channels, hidden_channels), ReLU())))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            conv.apply(init_weights)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adjs):
        x = x.to(torch.float)
        end_size = adjs[-1][-1][1]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.log_softmax(x, dim=-1)

    # Not implemented yet
    # @torch.no_grad()
    # def inference(self, x_all: torch.Tensor, device: torch.cuda.device,
    #               make_subgraph_iter: Callable[[torch.tensor],
    #                                            DeviceIterator]):
    #     return layerwise_inference(self, x_all, device, make_subgraph_iter)


class JKNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout=0.5, mode='max'):  # mode='cat'):
        conv_layer = SAGEConv
        kwargs = dict(bias=False)
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                conv_layer(hidden_channels, hidden_channels, **kwargs))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.jump = JumpingKnowledge(mode=mode, channels=hidden_channels,
                                     num_layers=num_layers)
        if mode == 'cat':
            self.lin = Linear(num_layers * hidden_channels, out_channels)
        else:
            self.lin = Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        self.jump.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj_t):
        x = x.to(torch.float)
        xs = []
        end_size = adj_t[-1][-1][1]
        for i, (edge_index, _, size) in enumerate(adj_t):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x[:end_size]]
        x = self.jump(xs)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        kwargs = dict(normalize=False, bias=False, improved=False)

        conv_layer = GCNConv
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.bns = torch.nn.ModuleList()

        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(
                hidden_channels, hidden_channels, **kwargs))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(conv_layer(
            hidden_channels, hidden_channels, **kwargs))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            conv.apply(init_weights)

    def forward(self, x, adjs):
        x = x.to(torch.float)
        end_size = adjs[-1][-1][1]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return torch.log_softmax(x, dim=-1)


class ARMA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        kwargs = dict(dropout=0.5)

        num_stacks = 1
        num_arma_layers = 1
        shared_weights = False

        conv_layer = ARMAConv
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.convs.append(conv_layer(
            in_channels, hidden_channels, num_stacks, num_arma_layers,
            shared_weights, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(
                hidden_channels, hidden_channels, num_stacks, num_arma_layers,
                shared_weights, **kwargs))
        self.convs.append(conv_layer(
            hidden_channels, hidden_channels, num_stacks, num_arma_layers,
            shared_weights, **kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            conv.apply(init_weights)

    def forward(self, x, adjs):
        x = x.to(torch.float)
        end_size = adjs[-1][-1][1]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return torch.log_softmax(x, dim=-1)

