# Getting information about the type of data we are dealing with

from rdkit.Chem import rdmolops
import os
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

def moltosvg(mol, molSize = (300,300), kekulize = True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def get_components(mol):
    mol_frags = rdmolops.GetMolFrags(mol, asMols = True)
    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    return len(mol_frags), len(largest_mol.GetAtoms())

#mol = Chem.MolFromPDBFile("/Users/padr/repos/linking/datasets/raw/refined-set/1a1e/1a1e_pocket.pdb")
#SVG(moltosvg(mol))

#graph = mol_to_complete_graph(mol, explicit_hydrogens=False, node_featurizer=CanonicalAtomFeaturizer, edge_featurizer=CanonicalBondFeaturizer)

from rdkit import RDLogger
#RDLogger.DisableLog('rdApp.*')
bad_data = ["1g7v", "1r1h", "2a5b", "2zjw", "1cps", "4abd"]
files_to_process = []
for path, dirs, files in os.walk('/Users/padr/repos/linking/datasets/raw/refined-set'):
    for file in files:
        if not (file.endswith('_ligand.mol2') or file.endswith('_ligand.sdf') or file.endswith('_pocket.pdb') or file.endswith('_protein.pdb')):
            print(file)
        if file.endswith('.mol2') and not file.split("_")[0] in bad_data:
            full_path = path + os.sep + file
            files_to_process.append(full_path)

graphs = []
total = len(files_to_process)
print("Starting to process " + str(total) + " files...")
i = 0
total_atoms = 0
good_atoms = 0
atoms = {"C"}
for path in sorted(files_to_process):
    i += 1
    mol = Chem.MolFromMol2File(path)
    try:
        frags = get_components(mol)
        total_atoms += frags[1]
        good_atoms += 1

        for atom in mol.GetAtoms():
            type = atom.GetSymbol()
            atoms.add(type)

        print(
            "(" + str(int(100 * i / total)) + "%) File " + os.path.basename(
                path) + " has X fragments and the largest is of size Y: " + str(frags)
        )
    except:
        print(
            "(" + str(int(100 * i / total)) + "%) File " + os.path.basename(
                path) + " has an exception."
        )
print("Average number of atoms in a ligand: " + str(total_atoms/good_atoms))
print("Set of atom types:")
print(atoms)