import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class LinearProj(nn.Module):
    """Implementation of the linear projection layer for GATC.
        Params:
            node_feat_dim: input node feature dimension.
            d_model: output dimension.
    """
    def __init__(self, node_feat_dim, d_model):
        super(LinearProj, self).__init__()

        self.proj = nn.Linear(node_feat_dim, d_model)

    def forward(self, x):
        return self.proj(x)
    
class GATC(nn.Module):
    """Implementation of the transformer heads module from the `"Graph Attention Networks"
        <https://arxiv.org/abs/1710.10903>` paper.
        Params:
            node_feat_dim: input node feature dimension.
            d_model: output dimension of linear projection.
            edge_dim: edge feature dimension.
            heads: total head. Default: 4
            project_edge_dim: projection of edge dimension
            more_skip: whether to use skip connection. Default: True
            last_average: whether to average the multi-head attentions. Default: False
    """
    def __init__(self, node_feat_dim, d_model, edge_dim, heads=4, project_edge_dim=None, more_skip=True, last_average=False):
        super().__init__()
        self.lp = LinearProj(node_feat_dim, d_model)
        self.more_skip = more_skip
        self.project_edge_dim = project_edge_dim
        if self.project_edge_dim is not None:
            self.lp_edge_attr = nn.Linear(edge_dim, project_edge_dim)
            edge_dim = project_edge_dim

        self.conv1 = GATConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean')
        
        self.conv2 = GATConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean')
        
        if last_average:
            self.conv3 = GATConv(d_model, d_model, heads, concat=False, edge_dim=edge_dim, aggr='mean')
        else:
            self.conv3 = GATConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean')

    def forward(self, data):
        x = self.lp(data.x)
        if self.project_edge_dim is not None:
            e = self.lp_edge_attr(data.edge_attr)
        else:
            e = data.edge_attr

        if self.more_skip:
            x = F.relu(x + self.conv1(x, data.edge_index, e))
            x = F.relu(x + self.conv2(x, data.edge_index, e))
            x = F.relu(x + self.conv3(x, data.edge_index, e))
        else:
            x = F.relu(self.conv1(x, data.edge_index, e))
            x = F.relu(self.conv2(x, data.edge_index, e))
            x = F.relu(self.conv3(x, data.edge_index, e))
        return x