#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean
import numpy as np
from scipy.sparse import csr_matrix

from Bio.PDB import PDBParser,ShrakeRupley
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm

from kan import KANLayer


# In[2]:


def get_surface_mask(structure, residues, sasa_threshold=1.0):
    """shape=[num_residues]"""
    sr = ShrakeRupley()
    sr.compute(structure, level="R")  # SASA
    
    surface_mask = []
    for res in residues:
        sasa = res.sasa if hasattr(res, "sasa") else 0.0
        surface_mask.append(sasa > sasa_threshold)
    return np.array(surface_mask)


# In[3]:


def parse_pdb(pdb_file, surface_sasa_threshold=1.0):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    residues = [res for res in structure.get_residues() if res.get_id()[0] == ' ']

    surface_mask = get_surface_mask(structure, residues)

    residues = [res for res, keep in zip(residues, surface_mask) if keep]
    if len(residues) == 0:
        raise ValueError("No surface residues detected! Try lowering sasa_threshold.")

    coords = np.array([res['CA'].get_coord() for res in residues])
    amino_acids = [res.get_resname() for res in residues]
    num_residues = len(coords)

    dist_matrix = cdist(coords, coords, metric='euclidean')

    connected = (dist_matrix < 20.0).astype(float)
    connected = csr_matrix(connected)

    angles = []
    for i in range(num_residues):
        angle = 0.0
        if 0 < i < num_residues - 1:
            v1 = coords[i] - coords[i - 1]
            v2 = coords[i] - coords[i + 1]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                v1 /= np.linalg.norm(v1)
                v2 /= np.linalg.norm(v2)
                dp = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(dp)
        angles.append(angle)

    return coords, amino_acids, dist_matrix, connected, angles


# In[4]:


def create_graph(coords, amino_acids, dist_matrix, connected, angles):
    # one-hot
    amino_acid_to_idx = {aa: i for i, aa in enumerate(set(amino_acids))}
    node_features = torch.tensor([amino_acid_to_idx[aa] for aa in amino_acids], dtype=torch.long)
    node_features = F.one_hot(node_features, num_classes=20).float()

    edge_index = connected.nonzero()
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_attr = []
    for i, j in edge_index.t():
        dist = dist_matrix[i, j]
        inv_dist_sq = 1.0 / (dist ** 2 + 1e-6)
        is_connected = connected[i, j]
        inv_dist_sq = is_connected*inv_dist_sq
        angle = 0
        if i-j == 1:
            angle = angles[i]
        if i-j == -1:
            angle = math.pi - angles[i]
        edge_attr.append([inv_dist_sq, is_connected, angle])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data
#X:[num_residues,20]
#edge_index：[2,connect]
#edge_attr：[connect，3]


# In[5]:

pdb_files_dir = '../result/target/stru'
output_dir = 'pdb_surf_graph_data'
os.makedirs(output_dir, exist_ok=True)

for pdb_file in tqdm(os.listdir(pdb_files_dir)):
    if pdb_file.endswith('.pdb'):
        pdb_path = os.path.join(pdb_files_dir, pdb_file)
        coords, amino_acids, dist_matrix, connected, angles = parse_pdb(pdb_path)
        graph_data = create_graph(coords, amino_acids, dist_matrix, connected, angles)

        output_file = os.path.join(output_dir, f'{os.path.splitext(pdb_file)[0]}_surf.pt')
        torch.save(graph_data, output_file)

print(f'Saved graph data for {pdb_file} to {output_file}')


# In[ ]:


graph_data_dir = 'pdb_surf_graph_data'

for graph_file in os.listdir(graph_data_dir):
    if graph_file.endswith('.pt'):
        graph_path = os.path.join(graph_data_dir, graph_file)
        graph_data = torch.load(graph_path)

        print(f"Loaded graph data from {graph_file}:")
        print(f"  - Number of nodes: {graph_data.num_nodes}")
        print(f"  - Number of edges: {graph_data.num_edges}")
        print(f"  - Node features shape: {graph_data.x.shape}")
        print(f"  - Edge index shape: {graph_data.edge_index.shape}")
        print(f"  - Edge features shape: {graph_data.edge_attr.shape}")
        print()


# In[ ]:


graph_list = []
for graph_file in os.listdir(graph_data_dir):
    if graph_file.endswith('.pt'):
        graph_path = os.path.join(graph_data_dir, graph_file)
        graph_data = torch.load(graph_path)
        graph_list.append(graph_data)

# DataLoader
batch_size = 32
loader = DataLoader(graph_list, batch_size=batch_size, shuffle=True)

for batch in loader:
    print(f"Batch size: {batch.num_graphs}")
    print(f"Batch node features shape: {batch.x.shape}")
    print(f"Batch edge index shape: {batch.edge_index.shape}")
    print(f"Batch edge features shape: {batch.edge_attr.shape}")





