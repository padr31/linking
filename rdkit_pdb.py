from rdkit import Chem
from rdkit.Chem import rdmolops

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

mol = Chem.MolFromFile("datasets/refined-set/1a1e/1a1e_ligand.mol2")

print(len(mol.GetBonds()))

'''for x in m.GetAtoms():
    print(x.GetIdx(), x.GetHybridization())
''' 
mol_frags = rdmolops.GetMolFrags(mol, asMols = True)
print(len(mol_frags))
largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
print(Chem.MolToSmiles(largest_mol))
print(len(largest_mol.GetAtoms()))