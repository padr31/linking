import torch
from biopandas.pdb import PandasPdb
from dgllife.utils import (CanonicalAtomFeaturizer, CanonicalBondFeaturizer,
                           mol_to_bigraph, mol_to_graph)
from rdkit import Chem
from torch_geometric import data

mol = Chem.MolFromPDBFile(
    "/Users/padr/repos/linking/datasets/raw/refined-set/1a1e/1a1e_pocket.pdb"
)

atoms = (
    PandasPdb()
    .read_pdb("/Users/padr/repos/linking/datasets/raw/refined-set/1a1e/1a1e_pocket.pdb")
    .df
)

g = mol_to_bigraph(
    mol,
    node_featurizer=CanonicalAtomFeaturizer(),
    edge_featurizer=CanonicalBondFeaturizer(),
)
print()

# Get node features from DGL graph and concatenate them
node_feature_names = g.node_attr_schemes().keys()
dgl_graph_features = [g.ndata[feat].float() for feat in node_feature_names]
dgl_graph_features = [
    f.unsqueeze(dim=1) if len(f.shape) == 1 else f for f in dgl_graph_features
]
node_features = torch.cat(dgl_graph_features, dim=1)

# Get edge features from DGL graph and concatenate them
edge_types = g.edge_attr_schemes().keys()
edge_feats = [g.edata[e].float() for e in edge_types]
edge_feats = [e.unsqueeze(dim=1) if len(e.shape) == 1 else e for e in edge_feats]
edge_feats = torch.cat(edge_feats, dim=1)

# Create the Torch Geometric graph
geom_graph = data.Data(
    x=node_features,
    edge_index=torch.stack(g.edges(), dim=0).long(),
    edge_attr=edge_feats,
)
print()
