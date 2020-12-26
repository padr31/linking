from graphein.construct_graphs import  ProteinGraph

# Initialise ProteinGraph class
pg = ProteinGraph(granularity='CA', insertions=False, keep_hets=False, intramolecular_interactions=None,
                  node_featuriser='meiler', get_contacts_path='/Users/padr/repos/linking/getcontacts',
                  pdb_dir='datasets/pdb/',
                  contacts_dir='/Users/padr/repos/linking/getcontacts',
                  exclude_waters=True, covalent_bonds=False, include_ss=True,include_ligand=False,
                  verbose=True, long_interaction_threshold=5, edge_distance_cutoff=10, edge_featuriser=None)

# graph = pg.dgl_graph_from_pdb_code('3eiy', chain_selection='all')

graph = pg.torch_geometric_graph_from_pdb_code("3eiy",
        chain_selection="all",
        edge_construction=["distance", "delaunay"],
        encoding=True,
        k_nn=None,)


print(graph.x)