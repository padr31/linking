import math

from ase.formula import Formula
from rdkit import DataStructs, Chem
from rdkit.Chem.Lipinski import NHOHCount, RingCount, NOCount
from rdkit.Chem import rdmolops, QED, Crippen
from rdkit.Geometry.rdGeometry import Point3D
from linking.util.encoding import to_bond_index, to_atom, allowable_atoms
from molgym.environment import MolecularEnvironment
from molgym.reward import InteractionReward
from molgym.spaces import ObservationSpace, ActionSpace
from rdkit.Chem import rdMolDescriptors
import os.path as op
import pickle

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


_fscores = None
def rdkit_sascore(m):
    def numBridgeheadsAndSpiro(mol, ri=None):
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return nBridgehead, nSpiro

    def readFragmentScores():
        global _fscores
        import gzip
        # generate the full path filename:
        data = pickle.load(gzip.open('./datasets/fpscores.pkl.gz'))
        outDict = {}
        for i in data:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        _fscores = outDict

    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore

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

def rdkit_logp(mol):
    return Crippen.MolLogP(mol)

def rdkit_fingerprint(mol):
    return Chem.RDKFingerprint(mol)

def rdkit_sanitize(mol, kekulize=True, addHs=False):
    m = Chem.Mol(mol.ToBinary())
    try:
        Chem.SanitizeMol(m)
    except:
        m = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(m)
        except:
            m = Chem.Mol(m.ToBinary())
    if addHs:
        try:
            m_hs = Chem.AddHs(m, addCoords=True)
        except:
            m_hs = Chem.Mol(m.ToBinary())
        m = m_hs
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

def tanimoto_score(ligand_mol, original_ligand_mol):
    ligand_fingerprint = rdkit_fingerprint(ligand_mol)
    original_fingerprint = rdkit_fingerprint(original_ligand_mol)
    return rdkit_tanimoto(ligand_fingerprint, original_fingerprint)

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