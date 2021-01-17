from __future__ import annotations

from torch import nn
from torch_geometric.nn import VGAE

from linking.config.config import Config
from linking.layers.gcn_encoders import VariationalGCNEncoder, GCNEncoder
from linking.model.base import BaseModel
from linking.layers.linear_encoders import LinearAtomLabelClassifier
from linking.model.model import MoleculeGenerator, SimpleModel

def build_model(config: Config):
    if config.model == "SimpleModel":
        return build_simple_model(config)
    elif config.model == "MoleculeGenerator":
        return build_generator_model(config)

def build_generator_model(config: Config) -> nn.Module:
    pocket_encoder = GCNEncoder(
        in_channels=config.pocket_encoder_in_channels, out_channels=config.pocket_encoder_out_channels)
    ligand_encoder = GCNEncoder(
        in_channels=config.ligand_encoder_in_channels, out_channels=config.ligand_encoder_out_channels)
    linear_atom_label_classifier = LinearAtomLabelClassifier(in_channels=config.ligand_encoder_out_channels, out_channels=config.num_allowable_atoms)

    model = MoleculeGenerator(pocket_encoder, ligand_encoder, linear_atom_label_classifier, None, None, config)
    return model

def build_simple_model(config: Config) -> nn.Module:
    pocket_encoder = GCNEncoder(
        in_channels=config.pocket_encoder_in_channels, out_channels=config.pocket_encoder_out_channels)
    ligand_encoder = GCNEncoder(
        in_channels=config.ligand_encoder_in_channels, out_channels=config.ligand_encoder_out_channels)

    model = SimpleModel(pocket_encoder, ligand_encoder)
    return model

