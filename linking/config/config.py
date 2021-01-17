from pydantic import BaseModel

class Config(BaseModel):
    # Core stuff
    root_dir: str = "/Users/padr/repos/linking/"
    dataset_root: str = "/Users/padr/repos/linking/datasets/"
    num_epochs: int = 3
    learning_rate: float = 0.0001

    # Data stuff
    train_test_ratio: int = 4
    num_train: int = 300

    # Model stuff
    model:str = "MoleculeGenerator"
    num_allowable_atoms = 46

    pocket_encoder_in_channels: int = 50
    pocket_encoder_out_channels: int = 50
    ligand_encoder_in_channels: int = 50
    ligand_encoder_out_channels: int = 50
    num_expansion_nodes: int = 10
    num_max_generated_atoms: int = 10
    num_max_generated_edges: int = 4*num_max_generated_atoms

    p_num_feats: int = 10
    m_num_feats: int = 10