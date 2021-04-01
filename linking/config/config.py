from pydantic import BaseModel

class Config(BaseModel):
    # Core stuff
    root_dir: str = "./"
    dataset: str = "pdb"
    dataset_root: str = "./datasets/"
    dataset_root_pdb: str = "./datasets/pdb/"
    dataset_root_dude: str = "./datasets/dude/"
    save_model: str = "./out_model/"
    logdir: str = "./out_logdir/"
    num_epochs: int = 10
    learning_rate: float = 0.003

    # Data stuff
    train_test_ratio: int = 10
    num_train: int = 2048
    eval_data: list = [0, 1, 2, 3, 4]
    num_eval_generate: int = 20

    # Model stuff
    model: str = "TeacherForcer"  # "MoleculeGenerator" # "SimpleModel"
    batch_size: int = 1
    num_allowable_atoms = 11
    num_allowable_bonds = 4

    pocket_encoder_in_channels: int = num_allowable_atoms+4
    pocket_encoder_out_channels: int = 50
    ligand_encoder_in_channels: int = num_allowable_atoms+4
    ligand_encoder_out_channels: int = 50
    graph_encoder_in_channels: int = num_allowable_atoms
    graph_encoder_out_channels: int = 50
    sch_net_hidden_channels: int = 64
