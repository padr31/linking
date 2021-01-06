from graphein.construct_graphs import ProteinGraph

# Initialise ProteinGraph class
pg = ProteinGraph(
    granularity="atom",
    insertions=False,
    keep_hets=False,
    intramolecular_interactions=None,
    get_contacts_path="/Users/padr/repos/linking/getcontacts",
    pdb_dir="datasets/pdb/",
    contacts_dir="/Users/padr/repos/linking/getcontacts",
    exclude_waters=True,
    covalent_bonds=False,
    include_ss=True,
    include_ligand=False,
    verbose=True,
    long_interaction_threshold=5,
    edge_distance_cutoff=10,
)

# graph = pg.dgl_graph_from_pdb_code('3eiy', chain_selection='all')

graph = pg._make_atom_graph(
    pdb_path="/Users/padr/repos/linking/datasets/raw/refined-set/1a1e/1a1e_pocket.pdb"
)

print(graph.x)
