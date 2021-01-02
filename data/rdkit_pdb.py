from dgllife.utils import (CanonicalAtomFeaturizer, CanonicalBondFeaturizer,
                           mol_to_bigraph, mol_to_complete_graph, mol_to_graph,
                           smiles_to_bigraph, smiles_to_complete_graph)
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D


def moltosvg(mol, molSize=(300, 300), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace("svg:", "")


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


mol = Chem.MolFromPDBFile(
    "/Users/padr/repos/linking/datasets/raw/refined-set/1g7v/1g7v_pocket.pdb"
)
with open("out.svg", "w") as svg_file:
    svg_file.write(moltosvg(mol))

print(len(mol.GetBonds()))

mol_frags = rdmolops.GetMolFrags(mol, asMols=True)
print(len(mol_frags))
largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
g = mol_to_bigraph(largest_mol)
print()
