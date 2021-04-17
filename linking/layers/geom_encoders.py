from linking.layers.schnet_custom import SchNet
import torch


class Sch(torch.nn.Module):
    def __init__(self, hidden_channels: int, output_channels: int):
        super(Sch, self).__init__()
        self.sch_layer = SchNet(hidden_channels=hidden_channels, output_channels=output_channels, cutoff=2.5, num_filters=32, num_interactions=3, num_gaussians=20)

    def forward(self, z, pos) -> torch.Tensor:
        return self.sch_layer(z, pos).relu()

