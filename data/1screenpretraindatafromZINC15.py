#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from rdkit import Chem
from tqdm import tqdm


# In[2]:

def build_smiles_dict(smiles_file):
    smiles_dict = set()
    with open(smiles_file, 'r') as f:
        for line in f:
            smiles = line.strip()
            smiles_dict.update(smiles)
    return smiles_dict

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.RemoveHs(mol)
            standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return standardized_smiles
        else:
            return None
    except:
        return None


# In[3]:


def filter_smiles(zinc_folder, smiles_dict, output_file):
    with open(output_file, 'w') as out_f:
        num = 0
        for root, _, files in os.walk(zinc_folder):
            for file in tqdm(files):
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    print("file_path:",file_path)
                    with open(file_path, 'r') as in_f:
                        for line in tqdm(in_f):
                            smiles = line.split()[0].strip()
                            standardized_smiles = standardize_smiles(smiles)
                            if standardized_smiles and 34 <= len(standardized_smiles) <= 74 and all(char in smiles_dict for char in standardized_smiles):
                                num+=1
                                out_f.write(standardized_smiles + '\n')
        print("pretrained data num:",num)


# In[4]:


smiles_file = 'smiles.txt'
zinc_folder = 'ZINC15'
output_file = 'pretrainedsmiles.txt'
smiles_dict = build_smiles_dict(smiles_file)
filter_smiles(zinc_folder, smiles_dict, output_file)




