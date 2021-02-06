from __future__ import annotations

from torch import nn
from torch_geometric.nn import VGAE

from linking.config.config import Config
from linking.layers.gcn_encoders import VariationalGCNEncoder, GCNEncoder
from linking.model.base import BaseModel
from linking.layers.linear_encoders import LinearAtomClassifier, LinearEdgeClassifier, LinearEdgeSelector, LinearScorePredictor
from linking.model.model import MoleculeGenerator, SimpleModel
from linking.model.model_teacher_forcer import TeacherForcer

def build_model(config: Config):
    if config.model == "SimpleModel":
        return build_simple_model(config)
    elif config.model == "MoleculeGenerator":
        return build_generator_model(config)
    elif config.model == "TeacherForcer":
        return build_forcer_model(config)

def build_forcer_model(config: Config) -> nn.Module:
    pocket_encoder = GCNEncoder(
        in_channels=config.pocket_encoder_in_channels, out_channels=config.pocket_encoder_out_channels)
    ligand_encoder = GCNEncoder(
        in_channels=config.ligand_encoder_in_channels, out_channels=config.ligand_encoder_out_channels)
    graph_encoder = GCNEncoder(
        in_channels=config.num_allowable_atoms, out_channels=config.ligand_encoder_out_channels
    )
    linear_atom_classifier = LinearAtomClassifier(in_channels=config.ligand_encoder_out_channels, out_channels=config.num_allowable_atoms)

    # [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, G]
    edge_feature_size = 1 + config.pocket_encoder_out_channels + config.ligand_encoder_out_channels + 2*(config.graph_encoder_out_channels + config.num_allowable_atoms) + config.graph_encoder_out_channels
    linear_edge_selector = LinearEdgeSelector(edge_feature_size)
    linear_edge_classifier = LinearEdgeClassifier(edge_feature_size)

    model = TeacherForcer(pocket_encoder, ligand_encoder, graph_encoder, linear_atom_classifier, linear_edge_selector, linear_edge_classifier, config)
    return model

def build_generator_model(config: Config) -> nn.Module:
    pocket_encoder = GCNEncoder(
        in_channels=config.pocket_encoder_in_channels, out_channels=config.pocket_encoder_out_channels)
    ligand_encoder = GCNEncoder(
        in_channels=config.ligand_encoder_in_channels, out_channels=config.ligand_encoder_out_channels)
    graph_encoder = GCNEncoder(
        in_channels=config.num_allowable_atoms, out_channels=config.ligand_encoder_out_channels
    )
    linear_atom_classifier = LinearAtomClassifier(in_channels=config.ligand_encoder_out_channels, out_channels=config.num_allowable_atoms)

    # [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, G]
    edge_feature_size = 1 + config.pocket_encoder_out_channels + config.ligand_encoder_out_channels + 2*(config.graph_encoder_out_channels + config.num_allowable_atoms) + config.graph_encoder_out_channels
    linear_edge_selector = LinearEdgeSelector(edge_feature_size)
    linear_edge_classifier = LinearEdgeClassifier(edge_feature_size)

    model = MoleculeGenerator(pocket_encoder, ligand_encoder, graph_encoder, linear_atom_classifier, linear_edge_selector, linear_edge_classifier, config)
    return model

def build_simple_model(config: Config) -> nn.Module:
    pocket_encoder = GCNEncoder(
        in_channels=config.pocket_encoder_in_channels, out_channels=config.pocket_encoder_out_channels)
    ligand_encoder = GCNEncoder(
        in_channels=config.ligand_encoder_in_channels, out_channels=config.ligand_encoder_out_channels)
    score_predictor = LinearScorePredictor(in_channels=config.pocket_encoder_out_channels+config.ligand_encoder_out_channels, )
    model = SimpleModel(pocket_encoder, ligand_encoder, score_predictor)
    return model

