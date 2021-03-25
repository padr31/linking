import numpy as np
from rdkit import DataStructs, Chem
from rdkit.Chem.Lipinski import NHOHCount, RingCount

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

def lipinski_nhoh_count(mol):
    return NHOHCount(mol)

def lipinski_ring_count(mol):
    return RingCount(mol)