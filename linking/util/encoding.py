from ase.formula import Formula
from rdkit import Chem
import torch
import numpy as np

allowable_atoms = ['C', 'F', 'N', 'Cl', 'O', 'I', 'P', 'Br', 'S', 'H', 'Stop']
allowable_ligand_bonds = ["1", "2", "3", "ar", "am"]

allowable_rdkit_bonds = [
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]

ligand_bond_to_one_hot = {
    "1": [1., 0., 0., 0.],
    "2": [0., 1., 0., 0.],
    "3": [0., 0., 1., 0.],
    "ar": [0., 0., 0., 1.],
    "am": [1., 0., 0., 0.],
}

pocket_bond_to_one_hot = {
    Chem.rdchem.BondType.AROMATIC: [0., 0., 0., 1.],
    Chem.rdchem.BondType.SINGLE: [1., 0., 0., 0.],
    Chem.rdchem.BondType.DOUBLE: [0., 1., 0., 0.],
    Chem.rdchem.BondType.TRIPLE: [0., 0., 1., 0.],
}

empty_bond = [0., 0., 0., 0.]

allowable_angles = [71, 60, 1]
allowable_dihedrals = [180, 120, 60, 0, -60, -120, -180]

def to_one_hot(x, allowable_set=None, mapping=None):
    """
    Function for one hot encoding
    :param x: value to one-hot
    :param allowable_set: set of options to encode
    :param mapping: exact mapping from x to one hot, preferred over allowable set
    :return: one-hot encoding as torch tensor
    """
    if not mapping is None:
        return mapping[x].copy()
    else:
        return [1 if x == s else 0 for s in allowable_set]


def to_atom(t):
    return allowable_atoms[int(torch.dot(t, torch.tensor(range(t.size()[0]), dtype=torch.float, device=t.device)).item())]

def to_bond_valency(t):
    t_s = t.squeeze()
    return [1, 2, 3, 2][int(torch.dot(t_s, torch.tensor(range(t_s.size()[0]), dtype=torch.float, device=t.device)).item())]

def to_bond_index(t):
    t_s = t.squeeze()
    return [1, 2, 3, 4][int(torch.dot(t_s, torch.tensor(range(t_s.size()[0]), dtype=torch.float, device=t.device)).item())]

def to_bond_symbol(t):
    return ['-', '=', ':=:', '..'][to_bond_index(t) - 1]

def to_bond_length(a1, a2, bond):
    key = to_atom(a1) + to_bond_symbol(bond) + to_atom(a2)
    x = allowable_bond_lengths[key] if key in allowable_bond_lengths else allowable_bond_lengths['arbitrary']
    return torch.tensor(x, device=a1.device)

def to_angle(t):
    return allowable_angles[int(torch.dot(t, torch.tensor(range(t.size()[0]), dtype=torch.float, device=t.device)).item())]

def to_dihedral(t):
    return allowable_dihedrals[int(torch.dot(t, torch.tensor(range(t.size()[0]), dtype=torch.float, device=t.device)).item())]

def closest(a, arr):
    return min(arr, key=lambda x: abs(x - a))

def encode_angle(angle, device=None):
    discretised_angle = closest(180*angle/np.pi, allowable_angles)
    return torch.tensor(to_one_hot(discretised_angle, allowable_angles), device=device, dtype=torch.float)

def encode_dihedral(dihedral, device=None):
    discretised_dihedral = closest(180*dihedral/np.pi, allowable_dihedrals)
    return torch.tensor(to_one_hot(discretised_dihedral, allowable_dihedrals), device=device, dtype=torch.float)

from ase.data import atomic_numbers

def to_molgym_action_type(atom_label: torch.Tensor, atom_coords: torch.Tensor):
    atomic_number = atomic_numbers[to_atom(atom_label)]
    position = (atom_coords[0].item(), atom_coords[1].item(), atom_coords[2].item())

    return (atomic_number, position)

def ligand_to_molgym_formula(ligand_atoms: torch.Tensor):
    atoms_dict = {}
    atoms_list = [to_atom(one_hot_atom_label) for one_hot_atom_label in ligand_atoms]
    for atom_symbol in atoms_list:
        if atom_symbol in atoms_dict:
            atoms_dict[atom_symbol] += 1
        else:
            atoms_dict[atom_symbol] = 1
    formula = Formula.from_dict(atoms_dict)
    assert formula.__eq__(Formula.from_list(atoms_list))
    return formula

def molgym_formula_to_ligand(formula: Formula, device):
    atoms_dict = formula.count()
    atoms_list = []
    for k, v in atoms_dict.items():
        atoms_list.extend([k] * v)
    atoms_list = list(map(lambda atom_symbol: to_one_hot(atom_symbol, allowable_atoms), atoms_list))
    ids_list = range(0, len(atoms_list))
    coords_list = [[0., 0., 0.]]*len(atoms_list)
    return torch.cat([
        torch.tensor(ids_list, device=device).unsqueeze(1),
        torch.tensor(coords_list, device=device),
        torch.tensor(atoms_list, device=device)
    ], dim=1)

allowable_bond_lengths = {
"C-N":1.4107755,
"C-C":1.5150901,
"C..O":1.255712,
"C=O":1.2334914,
"N-C":1.4107754,
"C-S":1.7459718,
"S-C":1.7459718,
"C..C":1.4013792,
"N-H":1.0122557,
"C-H":1.095748,
"O..C":1.255712,
"O=C":1.2334915,
"H-N":1.0122557,
"H-C":1.095748,
"C=C":1.3827366,
"C-O":1.4193139,
"O-P":1.603689,
"P..O":1.5154448,
"O-C":1.4193139,
"P-O":1.603689,
"O..P":1.5154448,
"P-N":1.6737605,
"C=N":1.3358338,
"O-H":0.9500009,
"N-P":1.6737605,
"N=C":1.3358338,
"H-O":0.9500009,
"S=O":1.4604245,
"N..C":1.3490263,
"C..N":1.3490263,
"N-S":1.6302401,
"O=S":1.4604245,
"S-N":1.6302402,
"N-N":1.3555375,
"N=O":1.2924565,
"O=N":1.2924569,
"C-F":1.3404396,
"C-Cl":1.7358768,
"F-C":1.3404396,
"Cl-C":1.7358768,
"P-C":1.8017883,
"C-P":1.8017882,
"S-H":1.008008,
"H-S":1.008008,
"N-O":1.374657,
"O-N":1.374657,
"C-I":2.1217341,
"I-C":2.1217341,
"C-Br":1.897452,
"Br-C":1.897452,
"S..O":1.4803548,
"O..S":1.4803548,
"P=O":1.5383024,
"O=P":1.5383024,
"N=N":1.3202432,
"C:=:N":1.1414758,
"N:=:C":1.1414758,
"C:=:C":1.1998755,
"S=N":1.5981185,
"N=S":1.5981185,
"S-S":2.046078,
"O-S":1.5391866,
"S-O":1.5391866,
"P=S":1.9899822,
"S=P":1.9899822,
"N..N":1.3403707,
"C=S":1.6924675,
"S=C":1.6924675,
"P-H":1.0079916,
"H-P":1.0079916,
"P-S":2.0013974,
"S-P":2.0013974,
"N:=:N":1.1924888,
"arbitrary": 1.5150901,
}