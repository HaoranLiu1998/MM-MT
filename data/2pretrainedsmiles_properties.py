#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm


input_file = "../data/pretrainedsmiles.txt"
output_file = "pretrainedsmiles_properties.txt"

# SMARTS：（-NH2）
amino_smarts = "[NX3;H2]"
amino_pattern = Chem.MolFromSmarts(amino_smarts)

with open(input_csv, "r", encoding="utf-8") as fin, open(output_txt, "w", encoding="utf-8") as fout:
    reader = csv.reader(fin)
    header = next(reader)
    smiles_idx = 0

    # 写入表头
    fout.write(
        "MolWt\tHBD\tHBA\tRotatableBonds\tLogP\tNumRings\tMaxRingSize\tNumAmino\n"
    )

    for row in tqdm(reader, desc="Processing SMILES"):
        smi = row[smiles_idx].strip()

        if not smi:
            fout.write("-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\n")
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fout.write("-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\n")
            continue

        mol_weight = Descriptors.MolWt(mol)

        hbd = rdMolDescriptors.CalcNumHBD(mol)

        hba = rdMolDescriptors.CalcNumHBA(mol)

        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        logp = Descriptors.MolLogP(mol)

        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()

        if num_rings > 0:
            ring_sizes = [len(r) for r in ring_info.AtomRings()]
            max_ring_size = max(ring_sizes)
        else:
            max_ring_size = 0

        num_amino = len(mol.GetSubstructMatches(amino_pattern))

        fout.write(
            f"{mol_weight:.2f}\t{hbd}\t{hba}\t{rotatable_bonds}\t{logp:.2f}\t{num_rings}\t{max_ring_size}\t{num_amino}\n"
        )

print("Mol 8 properties save:", output_txt)




