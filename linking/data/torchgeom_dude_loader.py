from __future__ import annotations
from torch_geometric.data import InMemoryDataset
from linking.data.data_util import pdb_file_to_torch_geometric, mol2_file_to_torch_geometric
from linking.util.encoding import allowable_atoms, ligand_bond_to_one_hot, pocket_bond_to_one_hot
from rdkit import Chem
from pathlib import Path
from typing import Dict, List
from rdkit.Chem.rdmolfiles import SDMolSupplier
from tqdm import tqdm
import pandas as pd
import os
import torch

bad_data = ['AA2AR', 'FGFR1', 'DEF', 'MCR', 'ROCK1', 'THB']

def create_dude_index_file(file):
    """
    Downloads overview table of DUD-E Dataset
    :param out_path: Path object to save file to
    """
    df = pd.read_html(
        "http://dude.docking.org/targets",
    )[1]
    df.to_csv((Path(file) / "dude_index.csv"))

def load_dude_index_file(file) -> pd.DataFrame:
    return pd.read_csv((Path(file) / "dude_index.csv"))


def load(config):
    """Loads DUD-E Dataset into Dataset objects"""
    # Load index file
    df = load_dude_index_file(config.dataset_root_dude)
    # Load dataset from index
    dataset = load_dude_dataset(config=config, df=df, types=["actives"])
    return dataset


def load_dude_dataset(
    config, df: pd.DataFrame, types=("actives", "decoys")
):
    """
    Specifies loading workflow for DUD-E

    Extract targets from index DF,
    load positive & negative mols,
    assign decoy status,
    return total dataset wrapper
    """

    print("Loading DUD-E dataset")
    # Extract target names
    target_names = df["Target Name"]

    dataframes = []
    for type in types:
        # Load positives into a dictionary indexed by target
        positive_mols = load_mols(
            dude_root_path=config.dataset_root_dude,
            target_names=target_names,
            bad_data=bad_data,
            type=type,
        )

        # Process dictionaries into dataframes
        mols = (
            pd.DataFrame.from_dict(positive_mols, orient="index")
            .T.unstack()
            .dropna()
            .reset_index(level=0, drop=False)
        )
        mols.columns = ["target", "mol"]
        # Assign decoy status to dataframe
        mols["decoy"] = 1 if type == "decoys" else 0
        dataframes.append(mols)

    # concatenate dataframes
    combi = pd.concat(dataframes)

    samples = {}
    samples["target"] = list(combi["target"])
    samples["mol"] = list(combi["mol"])
    samples["label"] = list(combi["decoy"])

    del dataframes
    del combi
    return samples

def load_mols(
    dude_root_path: Path, target_names: pd.Series, bad_data, type: str
) -> Dict[str, List[Chem.Mol]]:
    """
    Loads molecules in DUD-E from an SDFSupplier using RDKit. Positive and decoy ligands for each target are each stored
    in a SDF file containing all the molecules. We produce a dictionary indexed by the target name and containing
    a list of all the mol objects for that target.
    :param dude_root_path: Path object pointing to DUDE dataset root.
    :param target_names: pd.Series containing the target names
    :param type: str {"actives", "decoys"} indicating whether or not to load active or inactive molecules
    :return: Dictionary of targets and molecules
    """
    print(f"Loading {type} molecules...")

    target_mol_map = {}

    # Some molecules are broken/no files provided. We handle this here.
    for target in tqdm(target_names):
        if not target in bad_data:
            try:
                target_mol_map[target] = [
                    mol
                    for mol in SDMolSupplier((dude_root_path + "raw/" + target + f"/{type}_final.sdf"))
                ]
            except:
                print(f"{target} has broken mols. Possibly no SDF file provided in DUD-E")
                continue
    return target_mol_map


def process_dir(dir, file_ending, bad_data):
    # Read data into huge `Data` list.
    files_to_process = []
    for path, dirs, files in os.walk(dir):
        for file in files:
            # split -1 not good for ligands, but doesnt matter because they are not 1:1 with pockets in dude
            if file.endswith(file_ending) and not (path.split("/")[-1].upper() in bad_data or path.split("/")[-2].upper() in bad_data):
                full_path = path + os.sep + file
                files_to_process.append(full_path)

    return files_to_process

class DudeLigandDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DudeLigandDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["ligands.pt"]

    def process(self):
        # Read data into huge `Data` list. and save
        files_to_process = process_dir(
            self.raw_dir, "_actives_ligand.mol2", bad_data
        )

        graphs = []
        total = len(files_to_process)
        print("Starting to process " + str(total) + " files...")
        i = 0
        for path in sorted(files_to_process):
            i += 1
            print("(" + str(int(100 * i / total)) + "%) Processing " + os.path.basename(path))
            g = mol2_file_to_torch_geometric(path, allowable_atoms, ligand_bond_to_one_hot)

            # torchgeom_plot_3D(g, 90)
            graphs.append(g)

        if self.pre_filter is not None:
            graphs = [g for g in graphs if self.pre_filter(g)]

        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class DudePocketDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DudePocketDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["pockets.pt"]

    def process(self):
        # Read data into huge `Data` list. and save
        files_to_process = process_dir(
            self.raw_dir, "pocket.pdb", bad_data
        )

        graphs = []
        total = len(files_to_process)
        print("Starting to process " + str(total) + " files...")
        i = 0
        for path in sorted(files_to_process):
            i += 1
            print("(" + str(int(100 * i / total)) + "%) Processing " + path)
            g = pdb_file_to_torch_geometric(path, allowable_atoms, pocket_bond_to_one_hot)
            # torchgeom_plot_3D(g, 90)
            graphs.append(g)

        if self.pre_filter is not None:
            graphs = [g for g in graphs if self.pre_filter(g)]

        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

# DudePocketDataset(Config().dataset_root_dude)
# DudeLigandDataset(Config().dataset_root_dude)
# load(config=Config())
