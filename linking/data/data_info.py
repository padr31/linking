# Getting information about the type of data we are dealing with
from rdkit.Chem import rdmolops
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

from linking.data.data_util import parse_mol2_bonds
from linking.data.torchgeom_dude_loader import DudeLigandDataset
from linking.util.encoding import to_atom, to_bond_index, to_angle, encode_angle, to_dihedral, encode_dihedral
from linking.util.coords import calc_angle, calc_dihedral, calc_position, calc_angle_p, calc_dihedral_p
from linking.data.torchgeom_pdb_loader import PDBLigandDataset
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from linking.util.eval import to_rdkit, lipinski_nhoh_count, rdkit_sanitize, rdkit_sascore, rdkit_logp, qed_score
from linking.util.plotting import mol_to_svg


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def get_components(mol):
    mol_frags = rdmolops.GetMolFrags(mol, asMols = True)
    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    return len(mol_frags), len(largest_mol.GetAtoms())

files_to_process = []
def get_files_to_process():
    bad_data = ["1g7v", "1r1h", "2a5b", "2zjw", "1cps", "4abd"]
    for path, dirs, files in os.walk('/Users/padr/repos/linking/datasets/raw/refined-set'):
        for file in files:
            if not (file.endswith('_ligand.mol2') or file.endswith('_ligand.sdf') or file.endswith('_pocket.pdb') or file.endswith('_protein.pdb')):
                print(file)
            if file.endswith('ligand.mol2') and not file.split("_")[0] in bad_data:
                full_path = path + os.sep + file
                files_to_process.append(full_path)

def getAtomTypes():
    total = len(files_to_process)
    print("Starting to process " + str(total) + " files...")
    i = 0
    atoms = {"C"}
    for path in sorted(files_to_process):
        i += 1
        mol = Chem.MolFromPDBFile(path)
        try:
            for atom in mol.GetAtoms():
                type = atom.GetSymbol()
                atoms.add(type)

            print(
                "(" + str(int(100 * i / total)) + "%) File " + os.path.basename(
                   path) + " has X fragments and the largest is of size Y: "# + str(frags)
             )
        except:
            print(
                "(" + str(int(100 * i / total)) + "%) File " + os.path.basename(
                    path) + " has an exception."
            )
    #print("Average number of atoms in a ligand: " + str(total_atoms/good_atoms))
    print("Set of atom types:")
    print(atoms)

def getBondTypes():
    total = len(files_to_process)
    print("Starting to process " + str(total) + " files...")
    i = 0
    bonds = {'1'}
    for path in sorted(files_to_process):
        i += 1
        b = parse_mol2_bonds(path)
        l = list(b.loc[:, "bond_type"].unique())
        [bonds.add(x) for x in l]

        print(
            "(" + str(int(100 * i / total)) + "%) File " + os.path.basename(
                path)
        )

    # print("Average number of atoms in a ligand: " + str(total_atoms/good_atoms))
    print("Set of bond types:")
    print(bonds)

def getAtomCounts(dataset):
    atoms = {}
    atom_count = 0
    mol_count = 0
    for data in tqdm(dataset):
        mol_count += 1
        for atom in data.x:
            atom = to_atom(atom[4:])
            atom_count += 1
            if atom in atoms:
                atoms[atom] += 1
            else:
                atoms[atom] = 1
    for k,v in atoms.items():
        atoms[k] = v/mol_count
    print(atoms)
    print(atom_count)
    print(mol_count)

def getBondCounts(dataset):
    bonds = {}
    bond_count = 0
    mol_count = 0

    for data in tqdm(dataset):
        mol_count += 1
        for attr in data.edge_attr:
            bond = to_bond_index(attr)
            bond_count += 1
            if bond in bonds:
                bonds[bond] += 1
            else:
                bonds[bond] = 1
    for k,v in bonds.items():
        bonds[k] = v/mol_count
    print(bonds)
    print(bond_count)
    print(mol_count)

from rdkit.Chem.Lipinski import RingCount

def getRingCounts(dataset):
    rings = {}
    ring_count = 0
    mol_count = 0

    for data in tqdm(dataset):
        mol_count += 1
        rdkit_mol = to_rdkit(Data(x=data.x[:, 4:], edge_index=data.edge_index, edge_attr=data.edge_attr))
        # rdkit_mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(rdkit_mol)

        #ligand_svg = mol_to_svg(rdkit_mol)

        #with open("./out_svg/ligand_" + data.protein_name + ".svg", "w") as svg_file:
        #    svg_file.write(ligand_svg)

        rings_list = rdkit_mol.GetRingInfo().AtomRings()
        ring_count += RingCount(rdkit_mol)
        for r in rings_list:
            ring_type = str(len(r))
            if ring_type in rings:
                rings[ring_type] += 1
            else:
                rings[ring_type] = 1

    for k, v in rings.items():
        rings[k] = v / mol_count

    others = 0
    for k, v in rings.items():
        if int(k) > 7:
            others += v
    rings['>7'] = others

    print(rings)
    print(ring_count/mol_count)
    print(mol_count)

def getNoohCounts(dataset):
    nooh_count = 0
    mol_count = 0

    for data in tqdm(dataset):
        rdkit_mol = to_rdkit(Data(x=data.x[:, 4:], edge_index=data.edge_index, edge_attr=data.edge_attr))
        rdkit_mol = rdkit_sanitize(rdkit_mol)
        # rdkit_mol.UpdatePropertyCache()
        # Chem.GetSymmSSSR(rdkit_mol)
        nooh = lipinski_nhoh_count(rdkit_mol)
        if not nooh is None:
            mol_count += 1
            nooh_count += nooh

    print(nooh_count/mol_count)

def getNoohCounts(dataset):
    nooh_count = 0
    mol_count = 0

    for data in tqdm(dataset):
        rdkit_mol = to_rdkit(Data(x=data.x[:, 4:], edge_index=data.edge_index, edge_attr=data.edge_attr))
        rdkit_mol = rdkit_sanitize(rdkit_mol)
        # rdkit_mol.UpdatePropertyCache()
        # Chem.GetSymmSSSR(rdkit_mol)
        nooh = lipinski_nhoh_count(rdkit_mol)
        if not nooh is None:
            mol_count += 1
            nooh_count += nooh

    print(nooh_count/mol_count)

def getSasCounts(dataset):
    sas_count = 0
    mol_count = 0
    invalid = 0

    for data in tqdm(dataset):
        rdkit_mol = to_rdkit(Data(x=data.x[:, 4:], edge_index=data.edge_index, edge_attr=data.edge_attr))
        rdkit_mol = rdkit_sanitize(rdkit_mol)
        # rdkit_mol.UpdatePropertyCache()
        # Chem.GetSymmSSSR(rdkit_mol)
        try:
            sas = rdkit_sascore(rdkit_mol)
        except:
            invalid +=1
        if not sas is None:
            mol_count += 1
            sas_count += sas

    print(sas_count / mol_count)
    print(invalid)

def getLogPCounts(dataset):
    logp_count = 0
    mol_count = 0
    invalid = 0
    for data in tqdm(dataset):
        rdkit_mol = to_rdkit(Data(x=data.x[:, 4:], edge_index=data.edge_index, edge_attr=data.edge_attr))
        rdkit_mol = rdkit_sanitize(rdkit_mol)
        # rdkit_mol.UpdatePropertyCache()
        # Chem.GetSymmSSSR(rdkit_mol)
        try:
            logp = rdkit_logp(rdkit_mol)
        except:
            invalid +=1
        if not logp is None:
            mol_count += 1
            logp_count += logp

    print(logp_count / mol_count)
    print(invalid)

def getQEDCounts(dataset):
    qed_count = 0
    mol_count = 0
    invalid = 0
    for data in tqdm(dataset):
        rdkit_mol = to_rdkit(Data(x=data.x[:, 4:], edge_index=data.edge_index, edge_attr=data.edge_attr))
        rdkit_mol = rdkit_sanitize(rdkit_mol)
        # rdkit_mol.UpdatePropertyCache()
        # Chem.GetSymmSSSR(rdkit_mol)
        try:
            qed = qed_score(rdkit_mol)
        except:
            invalid +=1
        if not qed is None:
            mol_count += 1
            qed_count += qed

    print(qed_count / mol_count)
    print(invalid)


def getBfsAngleTypes(dataset):
    Q = []    # returns true if new things were added, i.e. we have new angles to calculate
    def qpush(edge):
        if (edge[0] == -1 or edge[1] == -1): # stop edge
            return False
        else:  # new edge
            Q.append(edge)
            if len(Q) == 4:
                Q.pop(0)
            assert len(Q) <= 3
            return True

    angles = {'0': 0}
    dyhedrals = {'0': 0}
    nan_count = 0

    for data in dataset:
        def edge_vec(edge):
            return data.x[edge[1]][1:4].numpy() - data.x[edge[0]][1:4].numpy()

        print("Pocessing " + str(data.name))
        Q = []
        for i in range(0, len(data.bfs_index)):
            have_new_node = qpush(data.bfs_index[i])
            queue_length = len(Q)
            if have_new_node and queue_length >= 2:  # new angle available
                vec = edge_vec(Q[queue_length-1])
                base_vec = edge_vec(Q[queue_length-2])
                angle = calc_angle(base_vec, vec)
                angle_degrees = (angle / np.pi) * 180
                if math.isnan(angle_degrees):
                    nan_count += 1
                    continue
                if str(int(angle_degrees)) in angles:
                    angles[str(int(angle_degrees))] += 1
                else:
                    angles[str(int(angle_degrees))] = 1
            if have_new_node and queue_length >= 3:  # new dyhedral available
                dihedral_base_vec = edge_vec(Q[queue_length-3])
                dih = calc_dihedral(dihedral_base_vec, base_vec, vec)

                p3 = data.x[Q[queue_length-1][0]][1:4].cpu().numpy()
                p4 = data.x[Q[queue_length-1][1]][1:4].cpu().numpy()
                new_pos = calc_position(dihedral_base_vec, base_vec, p3, np.linalg.norm(p4-p3), angle, dih)
                assert np.linalg.norm(p4 - new_pos) < 0.01
                dyhedral = (dih / np.pi) * 180
                if math.isnan(dyhedral):
                    nan_count += 1
                    continue
                if str(int(dyhedral)) in dyhedrals:
                    dyhedrals[str(int(dyhedral))] += 1
                else:
                    dyhedrals[str(int(dyhedral))] = 1

    for dic in [angles, dyhedrals]:
        dic = {k: v for k, v in dic.items() if v > 1000}
        sorted_tupples = sorted([(k, v) for k, v in dic.items()], key=lambda e: e[1])
        x = [e[0] for e in sorted_tupples]
        y = [e[1] for e in sorted_tupples]
        plt.bar(x, y, color='g', width=1)
        plt.xticks(rotation=-90)
        plt.show()

    print("Set of angle types:")
    print(angles)
    print("Set of dyhedral angle types:")
    print(dyhedrals)
    print('Nan count:')
    print(nan_count)

def getPointAngleTypes(dataset):
    angles = {'0': 0}
    dihedrals = {'0': 0}
    nan_count = 0
    num_mols = 0

    def min_dist_points(p, coords):
        sorted_c = sorted(enumerate(coords), key=lambda t: np.linalg.norm(p-t[1]))
        return coords[sorted_c[2][0]], coords[sorted_c[1][0]]

    for data in tqdm(dataset):
        num_mols += 1
        coords = np.ones((data.x.size(0), 3))
        coords[data.bfs_index[0][0]] = data.x[data.bfs_index[0][0], 1:4]
        for i in range(0, len(data.bfs_index)):
            from_i = data.bfs_index[i][0]
            to_i = data.bfs_index[i][1]
            if to_i == -1:
                continue
            to_c = data.x[to_i, 1:4]
            if np.linalg.norm(coords[to_i]-to_c.numpy()) == 0.0:
                continue
            try:
                min_dist_p = min_dist_points(coords[from_i], coords)
            except:
                print('not enough points to get angle')
                continue
            angle = to_angle(encode_angle(calc_angle_p(min_dist_p[1], coords[from_i], to_c)))
            dihedral = to_dihedral(encode_dihedral(calc_dihedral_p(min_dist_p[0], min_dist_p[1], coords[from_i], to_c)))
            coords[to_i] = to_c
            if str(angle) in angles:
                angles[str(angle)] += 1
            else:
                angles[str(angle)] = 1
            if math.isnan(dihedral):
                print('none')
            if str(dihedral) in dihedrals:
                dihedrals[str(dihedral)] += 1
            else:
                dihedrals[str(dihedral)] = 1

    for k, v in angles.items():
        angles[k] = v/num_mols

    for k, v in dihedrals.items():
        dihedrals[k] = v/num_mols

    for dic in [angles, dihedrals]:
        dic = {k: v for k, v in dic.items() if v > 1000}
        sorted_tupples = sorted([(k, v) for k, v in dic.items()], key=lambda e: e[1])
        x = [e[0] for e in sorted_tupples]
        y = [e[1] for e in sorted_tupples]
        plt.bar(x, y, color='g', width=1)
        plt.xticks(rotation=-90)
        plt.show()

    print("Set of angle types:")
    print(angles)
    print("Set of dyhedral angle types:")
    print(dihedrals)
    print(num_mols)

def getDistanceTypes(dataset):

    distances = {}

    for data in dataset:
        def edge_vec(edge):
            return data.x[edge[1]][1:4].numpy() - data.x[edge[0]][1:4].numpy()

        def bond_symbol(bond_index):
            return ['-', '=', ':=:', '..'][bond_index-1]

        print("Pocessing " + str(data.name))
        for i in range(0, len(data.edge_index[0])):
            edge = (data.edge_index[0][i], data.edge_index[1][i])
            bond = to_bond_index(data.edge_attr[i])
            bond_sym = bond_symbol(bond)
            atom1 = to_atom(data.x[edge[0]][4:])
            atom2 = to_atom(data.x[edge[1]][4:])
            vec = edge_vec(edge)
            dist = np.linalg.norm(vec)
            key = atom1 + bond_sym + atom2
            if not key in distances:
                distances[key] = []
            distances[key].append(dist)
    print("{")
    for key in distances:
        arr = np.array(distances[key])
        print('"' + key + '":' + str(np.mean(arr)) + ",")
        # print(key + " num: " + str(len(arr)) + ", mean-length: " + str(np.mean(arr)) + ", std: " + str(np.std(arr)))
    print("}")

d = PDBLigandDataset(root="/Users/padr/repos/linking/datasets/pdb")
#getAtomCounts(d)
#getBondCounts(d)
getRingCounts(d)
getNoohCounts(d)
getSasCounts(d)
getLogPCounts(d)
getQEDCounts(d)
#getPointAngleTypes(d)