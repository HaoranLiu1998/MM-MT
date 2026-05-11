#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import torch
import pickle
import gc


smi_prot_file = "./multidata/smi_prot.xlsx"
id2refseq_file = "./multidata/ID2Refseq.xlsx"
mrna_id_file = os.path.join("./multidata/gene_seq_data", "mRNA_ID.txt")
mrna_seq_file = os.path.join("./multidata/gene_seq_data", "mRNA_Seq.txt")
prot_emb_dir = "./multidata/prot_emb_data"
prot_graph_dir = "./multidata/prot_graph_data"
batch_size = 1000
output_dir = "./data/multidata/output_batches"
os.makedirs(output_dir, exist_ok=True)

smi_prot_df = pd.read_excel(smi_prot_file, header=None, usecols=[0, 1, 2])
smi_prot_df.columns = ['smiles', 'target1', 'target2']

id2refseq_df = pd.read_excel(id2refseq_file, header=None, usecols=[0, 1])
id2refseq_df.columns = ['uniprot', 'geneid']
id2refseq_df['uniprot'] = id2refseq_df['uniprot'].astype(str).str.strip()
id2refseq_df['geneid'] = id2refseq_df['geneid'].astype(str).str.strip()
uniprot_to_gene = dict(zip(id2refseq_df['uniprot'], id2refseq_df['geneid']))
del id2refseq_df
gc.collect()

with open(mrna_id_file, "r") as f:
    mrna_ids = [line.strip() for line in f]
with open(mrna_seq_file, "r") as f:
    mrna_seqs = [line.strip().upper() for line in f]

mrna_ids = [id_.strip() for id_ in mrna_ids]
gene_to_seq = dict(zip(mrna_ids, mrna_seqs))

del mrna_ids, mrna_seqs
gc.collect()

nuc_to_token = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

def convert_seq_to_tokens(seq):
    tokens = []
    for nuc in seq:
        if nuc not in nuc_to_token:
            return None
        tokens.append(nuc_to_token[nuc])
    return tokens

def process_row(row):
    smiles = row['smiles']
    target1 = str(row['target1']).strip()
    target2 = str(row['target2']).strip()
    
    if not target1 or not target2:
        return None

    target_info = []
    for target in [target1, target2]:
        geneid = uniprot_to_gene.get(target)
        if geneid is None:
            return None
        
        seq = gene_to_seq.get(geneid)
        if seq is None:
            return None
        token_seq = convert_seq_to_tokens(seq)
        if token_seq is None:
            return None

        emb_path = os.path.join(prot_emb_dir, f"{target}.npy")
        if not os.path.exists(emb_path):
            return None
        try:
            prot_emb = np.load(emb_path)
        except Exception:
            return None

        graph_path = os.path.join(prot_graph_dir, f"alphafold_{target}.pt")
        if not os.path.exists(graph_path):
            return None
        try:
            prot_graph = torch.load(graph_path, map_location=torch.device('cpu'))
        except Exception:
            return None
        
        target_info.append((token_seq, prot_emb, prot_graph))
    
    if len(target_info) != 2:
        return None

    return (smiles, 
            target_info[0][0], target_info[0][1], target_info[0][2],
            target_info[1][0], target_info[1][1], target_info[1][2])

batch = []
valid_count = 0
total_count = 0

for _, row in tqdm(smi_prot_df.iterrows()):
    total_count += 1
    record = process_row(row)
    if record is not None:
        valid_count += 1
        batch.append(record)

    if len(batch) >= batch_size:
        batch_filename = os.path.join(output_dir, f"batch_{valid_count // batch_size}.pkl")
        with open(batch_filename, "wb") as f:
            pickle.dump(batch, f)
        print(f"save {len(batch)} to {batch_filename}")
        batch = []
        gc.collect()

if batch:
    batch_filename = os.path.join(output_dir, f"batch_{(valid_count // batch_size) + 1}.pkl")
    with open(batch_filename, "wb") as f:
        pickle.dump(batch, f)
    print(f"save {len(batch)} to {batch_filename}")

print(f"process {total_count} drug data，total {valid_count}。")





