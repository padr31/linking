import abc

import torch


class BaseModel(abc.ABC):
    def __init__(self, mol_encoder, protein_encoder, decoder) -> None:
        self._mol_encoder = mol_encoder
        self._protein_encoder = protein_encoder
        self._decoder = decoder

    @abc.abstractmethod
    def forward(self, x) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def train(self, x) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def test(self, x) -> torch.Tensor:
        pass
