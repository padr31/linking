import os
import pandas as pd
from Bio.PDB import *
from pathlib import Path
from rdkit import Chem

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path("/Users/padr/repos/linking/datasets/dude/raw").resolve()
targets = [dir for dir in os.listdir(ROOT_DIR) if not dir.startswith(".")]


class DistanceSelect(Select):
    """Defines a selection class for writing a structure"""

    def __init__(self, distances):
        self.distances = distances

    def accept_residue(self, residue):
        # Accepts residue if it's present in the keys of our thresholded dict
        if residue.id[1] in self.distances.keys():
            return 1
        else:
            return 0

def calculate_distance(protein, ligand_coords: np.ndarray):
    """
    Get Dictionary of distances from CA atoms (alpha carbons) to ligand
    Computes distance to each ligand atom, then takes the minimum.
    """
    dist = {}

    for i, residue in enumerate(protein.get_residues()):
        for atom in residue.get_atoms():
            if atom.id == "CA":
                atom_dist = []
                for lig_atom in range(ligand_coords.shape[0]):
                    vector = atom.coord - ligand_coords[lig_atom, :]
                    atom_dist.append(np.sqrt(np.sum(vector * vector)))
                dist[residue.id[1]] = min(atom_dist)
    return dist

def extract_pocket(target, threshold: float = 8.0):
    # Load Protein and ligand
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure(ROOT_DIR / target / "receptor.pdb", ROOT_DIR / target / "receptor.pdb")
    ligand = Chem.MolFromMol2File(str((ROOT_DIR / target / "crystal_ligand.mol2").resolve()))

    # Get ligand atomic coords
    lig_coords = ligand.GetConformer().GetPositions()

    # Get distance between residues and ligand atoms
    distances = calculate_distance(protein, lig_coords)

    # Apply Filtering
    distances = {k: v for k, v in distances.items() if v < threshold}

    # Write structure
    pdbio = PDBIO()
    pdbio.set_structure(protein)
    pdbio.save(str(Path(ROOT_DIR / target / "pocket.pdb").resolve()), select=DistanceSelect(distances))

for target in tqdm(targets):
    problematic = []

    # Some molecules are broken. Eg. in aa2ar so we skip these
    try:
        extract_pocket(target, threshold=8.0)
    except:
        problematic.append(target)
        continue

print(problematic)


