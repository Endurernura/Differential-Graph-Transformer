import os
import pandas as pd
import numpy as np
from rdkit import Chem


#这个程序的唯一任务就是把信息从.csv里面读出来。

def load_smiles(n):
    dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir, 'Train_data', 'smiles.csv')
    df = pd.read_csv(file_path)

    smiles_1 = df.iloc[n, 2]
    smiles_2 = df.iloc[n, 5]
    label = df.iloc[n, 6]

    return smiles_1, smiles_2, label

def delete_random_bond(smiles, deletion_ratio=0.6):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles 
    
    editable_mol = Chem.EditableMol(mol)
    
    num_bonds = mol.GetNumBonds()
    if num_bonds == 0:
        return smiles 
    
    num_deletions = int(num_bonds * deletion_ratio)
    if num_deletions == 0:
        num_deletions = 1 
    
    bond_indices = list(range(num_bonds))
    selected_bonds = np.random.choice(bond_indices,  size=min(num_deletions, len(bond_indices)), replace=False)
    
    for bond_idx in sorted(selected_bonds, reverse=True):
        bond_idx_int = int(bond_idx)
        bond = mol.GetBondWithIdx(bond_idx_int)
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        editable_mol.RemoveBond(begin_atom, end_atom)
    
    new_mol = editable_mol.GetMol()
    return Chem.MolToSmiles(new_mol)


def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


if __name__ == "__main__":
    n = 4  #test only
    smiles_1, smiles_2 = load_smiles(n)
    print(f"smiles_1: {smiles_1}")
    print(f"smiles_2: {smiles_2}")
