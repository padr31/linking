from __future__ import annotations
from torch import nn
from linking.config.config import Config
from linking.layers.gcn_encoders import GCNEncoder, CGCEncoder, GATEncoder, VariationalGATEncoder
from linking.layers.geom_encoders import Sch
from linking.layers.linear_encoders import LinearAtomClassifier, LinearEdgeClassifier, LinearEdgeSelector, \
    MLP, LinearEdgeRowClassifier
from linking.model.model import MoleculeGenerator, SimpleModel
from linking.model.model_qed import QED
from linking.model.model_teacher_forcer import TeacherForcer

def build_model(config: Config, device):
    if config.model == "SimpleModel":
        return build_simple_model(config, device)
    elif config.model == "MoleculeGenerator":
        return build_generator_model(config, device)
    elif config.model == "TeacherForcer":
        return build_forcer_model(config, device)
    elif config.model == "QED":
        return build_qed_model(config, device)

def build_forcer_model(config: Config, device) -> nn.Module:
    pocket_encoder = GCNEncoder(
        in_channels=config.pocket_encoder_in_channels, out_channels=config.pocket_encoder_out_channels)
    ligand_encoder = GCNEncoder(
        in_channels=config.ligand_encoder_in_channels, out_channels=config.ligand_encoder_out_channels)
    graph_encoder = GCNEncoder(
        in_channels=config.num_allowable_atoms, out_channels=config.ligand_encoder_out_channels
    )
    linear_atom_classifier = LinearAtomClassifier(in_channels=config.ligand_encoder_out_channels, out_channels=config.num_allowable_atoms)

    # [t, z_pocket, z_u, l_u, z_v, l_v, H_t, H_init]
    edge_feature_size = 1 + config.pocket_encoder_out_channels + 2*(config.graph_encoder_out_channels + config.num_allowable_atoms) + (config.ligand_encoder_out_channels + config.num_allowable_atoms) + (config.graph_encoder_out_channels + config.num_allowable_atoms)
    linear_edge_selector = LinearEdgeSelector(edge_feature_size)
    # used previously redundantly to classify all edges
    # linear_edge_classifier = LinearEdgeClassifier(edge_feature_size, config.num_allowable_bonds)
    linear_edge_row_classifier = LinearEdgeRowClassifier(edge_feature_size, config.num_allowable_bonds)

    # [C_avg, C[u], C[v], z_v[u], lab_v[u], z_v[v], lab_v[v], edge_type]
    coords_feature_size = 3*(config.sch_net_output_channels) + 2*(config.graph_encoder_out_channels+config.num_allowable_atoms) + config.num_allowable_bonds
    sch = Sch(hidden_channels=config.sch_net_hidden_channels, output_channels=config.sch_net_output_channels)
    linear_edge_angle_classifier = LinearEdgeRowClassifier(coords_feature_size, config.num_allowable_angles)
    linear_edge_dihedral_classifier = LinearEdgeRowClassifier(coords_feature_size, config.num_allowable_dihedrals)

    model = TeacherForcer(
        pocket_encoder,
        ligand_encoder,
        graph_encoder,
        linear_atom_classifier,
        linear_edge_selector,
        linear_edge_row_classifier,
        sch,
        linear_edge_angle_classifier,
        linear_edge_dihedral_classifier,
        config, device)
    return model

def build_qed_model(config: Config, device) -> nn.Module:
    # pocket_encoder = GCNEncoder(
    #    in_channels=config.pocket_encoder_in_channels, out_channels=config.pocket_encoder_out_channels)
    # ligand_encoder = CGCEncoder(
    #    in_channels=config.ligand_encoder_in_channels, out_channels=config.ligand_encoder_out_channels, edge_dim=config.num_allowable_bonds)
    ligand_encoder = VariationalGATEncoder(in_channels=config.ligand_encoder_in_channels, out_channels=config.ligand_encoder_out_channels, hidden_layers=6)
    mlp = MLP(in_channels=config.ligand_encoder_out_channels, out_channels=1, hidden_layers=0)
    # sch = Sch(config.sch_net_hidden_channels)

    model = QED(
        ligand_encoder,
        mlp,
        config, device)
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
    score_predictor = MLP(in_channels=config.pocket_encoder_out_channels+config.ligand_encoder_out_channels)
    model = SimpleModel(pocket_encoder, ligand_encoder, score_predictor)
    return model

