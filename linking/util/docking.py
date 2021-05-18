from typing import List, Iterable
from rdkit import Chem
import os
import subprocess
import tempfile

from rdkit.Chem import AllChem


def parse_score_only(smina_stdout: Iterable[str]):
    terms = []
    for line in smina_stdout:
        if line.startswith('## Name'):
            _, _, *parsed_terms = line.split()
            terms = parsed_terms
            break

    assert terms, "No terms specified in smina's scoring function"
    terms = [term.replace(',', '_') for term in terms]

    current_mode = -1
    results = []
    for line in smina_stdout:
        if line.startswith('Affinity:'):
            results.append(dict())
            current_mode += 1
            results[current_mode] = {}
            _, affinity, _ = line.split()
            results[current_mode]['affinity'] = float(affinity)
        elif line.startswith('Intramolecular energy:'):
            _, _, energy = line.split()
            results[current_mode]['intramolecular_energy'] = float(energy)
        elif line.startswith('##') and not line.startswith('## Name'):
            _, _, *term_values = line.split()
            results[current_mode]['pre_weighting_terms'] = {
                term: float(value) for term, value in zip(terms, term_values)
            }

    assert current_mode + 1 == len(results)

    return results

def _exec_subprocess(command: List[str], timeout: int = None) -> List[str]:
    cmd = ' '.join([str(entry) for entry in command])

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=timeout)
        out, err, return_code = str(result.stdout, 'utf-8').split('\n'), str(result.stderr, 'utf-8'), result.returncode

        if return_code != 0:
            print('Docking failed with command "' + cmd + '", stderr: ' + err)
            raise ValueError('Docking failed')

        return out
    except subprocess.TimeoutExpired:
        print('Docking failed with command ' + cmd)
        raise ValueError('Docking timeout')

def embed_mol(molecule):
    conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=True, ignoreSmoothingFailures=True)

    if conf_id == -1:
        conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=False, ignoreSmoothingFailures=True)

    if conf_id == -1:
        return -1
    return molecule

def optimise_mol(molecule):
    try:
        for max_iterations in [200, 2000, 20000, 200000]:
            if AllChem.UFFOptimizeMolecule(molecule, maxIters=max_iterations) == 0:
                break
        else:
            raise ValueError('Structure optimization failure')
    except Exception as e:
        raise ValueError(e)
    return molecule


def mol_to_mol2_file(mol, output_filename, embed=False):
    try:
        molecule = Chem.AddHs(mol, addCoords=(not embed))
        if embed:
            molecule = embed_mol(molecule)
            if molecule == -1:
                return -1

        # optimise_mol(molecule)

        Chem.MolToMolFile(molecule, output_filename)

        command = f'obabel -imol {output_filename} -omol2 -O {output_filename}'
        openbabel_return_code = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL).returncode

        if openbabel_return_code != 0:
            raise ValueError(f'Failed to convert rdkit mol to .mol2')
    except Exception as e:
        print('unable to dock')
    return output_filename

def dock_mol2(ligand_path, protein_path, output_path, bounding_box):
    cmd = [
        'smina',
        '--receptor', os.path.abspath(protein_path),
        '--ligand', os.path.abspath(ligand_path),
        '--center_x',  bounding_box[0].item(),
        '--center_y', bounding_box[1].item(),
        '--center_z', bounding_box[2].item(),
        '--size_x', 10,
        '--size_y', 10,
        '--size_z', 10,
        '--exhaustiveness', 8,
        '--out', os.path.abspath(output_path),
        '--scoring', 'vinardo',
    ]

    return _exec_subprocess(cmd, timeout=500)


def score(ligand_mol, protein_path, label, embed=False, dock=False, bounding_box=None):
    '''
    :param ligand_mol: rdkit mol file of ligand
    :param protein_path: path to .pdb protein file
    :param label: file label for saving docked/embedded molecules
    :param embed: whether to assign coords using RDkit, otherwise assumes coords are present
    :param dock: whether to dock the protein (needs bounding_box), otherwise assumes it is docked
    :param bounding_box: needs to be present when dock=True
    :return:
    '''
    with open("./out_tmp/" + label + ".mol2", "w") as ligand_path:
        embedding_result = mol_to_mol2_file(ligand_mol, ligand_path.name, embed=embed)
        if embedding_result == -1:
            return

        if dock:
            # with tempfile.NamedTemporaryFile(suffix='.mol2') as docked_path:
            with open("./out_tmp/" + label + "ligand_docked.mol2", "w") as docked_path:
                dock_mol2(ligand_path.name, protein_path, docked_path.name, bounding_box=bounding_box)

                ligand_mol2_file = docked_path.name if dock else ligand_path.name
                command = [
                        'smina',
                        '--scoring', 'vinardo',
                        '-l', os.path.abspath(ligand_mol2_file),
                        '--score_only',
                        '-r', os.path.abspath(protein_path),
                    ]
                smina_stdout = _exec_subprocess(command)
                results = parse_score_only(smina_stdout)
                return results[0]['affinity']