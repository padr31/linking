import torch
from torch_geometric.nn import SchNet

class Sch(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super(Sch, self).__init__()
        self.sch_layer = SchNet(hidden_channels=hidden_channels)

    def forward(self, z, pos) -> torch.Tensor:
        return self.sch_layer(z, pos).relu()
