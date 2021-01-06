from __future__ import annotations

from torch_geometric.nn import VGAE

from linking.layers.gcn_encoders import VariationalGCNEncoder
from linking.model.base import BaseModel


def build_model(config: Config) -> nn.Module:
    # Todo parses config to build model
    m_encoder = VariationalGCNEncoder(
        in_channels=config.m_channels, out_channels=config.m_channels
    )

    model = VGAE(m_encoder)
    return model
