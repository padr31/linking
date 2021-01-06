import abc

import torch


class BaseModel(abc.ABC):
    def __init__(self, mol_encoder, protein_encoder, decoder) -> None:
        ...

    @abc.abstractmethod
    def forward(self, x) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def train(self, x) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def test(self, x) -> torch.Tensor:
        ...
