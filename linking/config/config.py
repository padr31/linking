from pydantic import BaseModel

class Config(BaseModel):
    # Core stuff
    root_dir: str = "./"
    dataset: str = "dude"
    dataset_root: str = "./datasets/"
    dataset_root_pdb: str = "./datasets/pdb/"
    dataset_root_dude: str = "./datasets/dude/"
    save_model: str = "./out_model/"
    logdir: str = "./out_logdir"
    logdir_qed: str = "./out_logdir_qed/"
    num_epochs: int = 20
    learning_rate: float = 0.003  # 0.003 best

    # Data stuff
    specific_pockets: list = None # ['abl1', 'ace', 'aces', 'ada', 'ada17']  # relevant for dude, when set, only trains on ligands for the selected protein
    ligands_per_pocket: int = 100
    eval_data: list = [0, 1, 2]
    num_eval_generate: int = 30
    remove_hydrogens: bool = False  # rerun data processing step of ligands when you set this
    molgym_eval: bool = False
    num_molgym_eval: int = 30
    molgym_eval_formulas: list = ['C2H2O2', 'CH3NO', 'CH4O', 'C3H5NO3', 'C4H7N', 'C3H8O', 'C7H8N2O2']

    # Model stuff
    num_train: int = 2048
    num_test: int = 16
    train_test_ratio: int = 10
    model: str = "TeacherForcer"  # "MoleculeGenerator" # "SimpleModel"
    coords: bool = False
    batch_size: int = 16
    num_allowable_atoms = 11
    num_allowable_bonds = 4
    num_allowable_angles = 3
    num_allowable_dihedrals = 7

    pocket_encoder_in_channels: int = num_allowable_atoms+4
    pocket_encoder_out_channels: int = 50
    ligand_encoder_in_channels: int = num_allowable_atoms+4
    ligand_encoder_out_channels: int = 100
    graph_encoder_in_channels: int = num_allowable_atoms
    graph_encoder_out_channels: int = 100  # 100 best
    # need divisible by two
    sch_net_hidden_channels: int = 32
    sch_net_output_channels: int = 16
