from pydantic import BaseModel


class PDBRegressionConfig(BaseModel):
    # Core stuff
    root_dir: str = "/Users/arianjamasb/github/linking/"
    num_epochs: int = 100
    learning_rate: float = 0.0001
    # Data stuff

    # Model stuff
    num_p_layers: int = 2
    num_m_layers: int = 2
    p_out_channels: int = 16
    m_out_channels: int = 16
    p_num_feats: int = 10
    m_num_feats: int = 10
