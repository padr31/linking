from pydantic import BaseModel

class Config(BaseModel):
    # Core stuff
    root_dir: str = "./"
    dataset: str = "pdb"
    dataset_root: str = "./datasets/"
    dataset_root_pdb: str = "./datasets/pdb/"
    dataset_root_dude: str = "./datasets/dude/"
    num_epochs: int = 40
    learning_rate: float = 0.003

    # Data stuff
    train_test_ratio: int = 4
    num_train: int = 512

    # Model stuff
    model: str = "TeacherForcer"  # "MoleculeGenerator" # "SimpleModel"
    batch_size: int = 64
    num_allowable_atoms = 11

    pocket_encoder_in_channels: int = num_allowable_atoms+4
    pocket_encoder_out_channels: int = 50
    ligand_encoder_in_channels: int = num_allowable_atoms+4
    ligand_encoder_out_channels: int = 50
    graph_encoder_in_channels: int = num_allowable_atoms
    graph_encoder_out_channels: int = 50
    num_expansion_nodes: int = 50
    num_max_generated_atoms: int = 30
    num_max_generated_edges: int = 4*num_max_generated_atoms

    p_num_feats: int = 10
    m_num_feats: int = 10