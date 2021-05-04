from ase.formula import Formula
from rdkit import DataStructs, Chem
from rdkit.Chem.Lipinski import NHOHCount, RingCount, NOCount
from rdkit.Chem import rdmolops, QED
from rdkit.Geometry.rdGeometry import Point3D
from linking.util.encoding import to_bond_index, to_atom, allowable_atoms
from molgym.environment import MolecularEnvironment
from molgym.reward import InteractionReward
from molgym.spaces import ObservationSpace, ActionSpace


def to_rdkit(data, device=None):
    has_pos = hasattr(data, 'pos') and (not data.pos is None)
    node_list = []
    for i in range(data.x.size()[0]):
        node_list.append(to_atom(data.x[i]))

    # create empty editable mol object
    mol = Chem.RWMol()
    # add atoms to mol and keep track of index
    node_to_idx = {}
    invalid_idx = set([])
    for i in range(len(node_list)):
        if node_list[i] == 'Stop' or node_list[i] == 'H':
            invalid_idx.add(i)
            continue
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    added_bonds = set([])
    for i in range(0, data.edge_index.size()[1]):
        ix = data.edge_index[0][i].item()
        iy = data.edge_index[1][i].item()
        bond = to_bond_index(data.edge_attr[i])
        # add bonds between adjacent atoms
        if (str((ix, iy)) in added_bonds) or (str((iy, ix)) in added_bonds) or (iy in invalid_idx or ix in invalid_idx):
            continue
        # add relevant bond type (there are many more of these)
        if bond == 0:
            continue
        elif bond == 1:
            bond_type = Chem.rdchem.BondType.SINGLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 2:
            bond_type = Chem.rdchem.BondType.DOUBLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 3:
            bond_type = Chem.rdchem.BondType.TRIPLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        elif bond == 4:
            bond_type = Chem.rdchem.BondType.SINGLE
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        added_bonds.add(str((ix, iy)))

    if has_pos:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(data.pos.size(0)):
            if i in invalid_idx:
                continue
            p = Point3D(data.pos[i][0].item(), data.pos[i][1].item(), data.pos[i][2].item())
            conf.SetAtomPosition(node_to_idx[i], p)
        conf.SetId(0)
        mol.AddConformer(conf)

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    mol_frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    return largest_mol

def tanimoto(vec_a, vec_b):
    ab_common = vec_a.dot(vec_b)
    a_size = vec_a.dot(vec_a)
    b_size = vec_b.dot(vec_b)
    return ab_common / (a_size + b_size - ab_common)

def dice(vec_a, vec_b):
    ab_common = vec_a.dot(vec_b)
    a_size = vec_a.dot(vec_a)
    b_size = vec_b.dot(vec_b)
    return 2*ab_common / (a_size + b_size)

def rdkit_tanimoto(vec_a, vec_b):
    return DataStructs.FingerprintSimilarity(vec_a, vec_b, metric=DataStructs.TanimotoSimilarity)

def rdkit_fingerprint(mol):
    return Chem.RDKFingerprint(mol)

def rdkit_sanitize(mol, kekulize=True):
    m = Chem.Mol(mol.ToBinary())
    try:
        Chem.SanitizeMol(m)
    except:
        m = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(m)
        except:
            m = Chem.Mol(mol.ToBinary())
    return m

def lipinski_nhoh_count(mol):
    try:
        return NHOHCount(mol)
    except:
        return None

def lipinski_ring_count(mol):
    try:
        return RingCount(mol)
    except:
        return None

def lipinski(mol):
    try:
        if NHOHCount(mol) > 5:
            return False
        elif NOCount(mol) > 10:
            return False
        else:
            return True
    except:
        return False

def qed_score(mol):
    try:
        return QED.qed(mol)
    except:
        return None

def construct_molgym_environment(num_atoms, formulas):
    reward = InteractionReward()
    symbols = allowable_atoms[0:-1]
    observation_space = ObservationSpace(canvas_size=num_atoms, symbols=symbols)
    action_space = ActionSpace()
    env = MolecularEnvironment(reward=reward,
                               observation_space=observation_space,
                               action_space=action_space,
                               formulas=formulas)
    return env