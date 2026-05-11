# Geometric Pocket-Centric Protein Encoding for Polypharmacology-Guided Multi-Target Drug Design

Official PyTorch implementation of the paper **“Geometric Pocket-Centric Protein Encoding for Polypharmacology-Guided Multi-Target Drug Design”**, accepted at ICML 2026.

---

## Abstract

Polypharmacology provides a powerful strategy for treating complex diseases, but identifying molecules that simultaneously satisfy coupled constraints across multiple biological targets remains difficult. Existing methods typically model protein pockets in isolation and struggle to jointly account for multiple heterogeneous binding sites when designing a single shared ligand. To address these limitations, we propose a pocket-structure-centric generative framework for polypharmacology. This framework introduces a novel protein topological representation that selectively masks ligand-irrelevant residues while explicitly modeling backbone folding geometry and inter-residue spatial proximity within binding pockets. In addition, structural representations are jointly fused with amino acid and nucleotide sequences to capture their complementary information across targets. Experiments on COVID-19, schizophrenia, and tumor targets show that this framework generates valid candidates with significantly improved binding affinities compared to state-of-the-art methods.

---

# Install Environment

## 1. GPU Environment

CUDA 12.6

---

## 2. Create Conda Environment

```bash
conda create -n polypharma python=3.11.9

conda activate polypharma
```

---

## 3. Install Required Packages

```bash
conda install -c rdkit rdkit
```

Install PyTorch and related dependencies according to your CUDA version.

Example:

```bash
pip install torch torchvision torchaudio
```

Install additional dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

## 1. Pretraining Dataset

The pretraining dataset should be downloaded from ZINC15.

### Filtering Criteria

Please filter molecules satisfying:

- LogP ≤ 3
- Molecular Weight ≤ 300

The final dataset contains more than 80 million molecules.

Save the processed dataset into:

```bash
./data/ZINC15
```

---

## 2. Multi-Target Training Dataset

The processed multi-target dataset is provided in:

```bash
./data/multidata/filtered_dual_target_data2.csv
```

The CSV file contains:

```text
SMILES
Target1_Uniprot
Target1_Encoded_Gene_Seq
Target2_Uniprot
Target2_Encoded_Gene_Seq
```

---

## 3. Protein Structure Data

Protein pocket structures are stored in:

```bash
./data/multidata/stru_data
```

---

## 4. Protein Embedding Features

Protein embeddings processed by ProTrans are stored in:

```bash
./data/multidata/prot_emb_data
```

---

# Data Preprocessing

## 1. Pretraining Data Preprocessing

Script:

```bash
./data/1screenpretraindatafromZINC15.py
```

This script preprocesses molecular data downloaded from ZINC15.

---

## 2. Molecular Property Calculation

Script:

```bash
./data/2pretrainedsmiles_properties.py
```

This script calculates eight physicochemical properties for molecular pretraining.

---

## 3. Protein Structure Graph Construction

Script:

```bash
./data/3PDB2graph.py
```

This script converts protein PDB structures into branch topological graph representations.

---

## 4. Multi-Modal Multi-Target Dataset Construction

Script:

```bash
./data/4screenmultimodaldata.py
```

This script preprocesses multi-modal, multi-target, and multi-property training data.

---

# Model Training

## 1. Molecular Pretraining

Script:

```bash
./model/1pretrainedmodelv1-MutiMolGPT.py
```

Example:

```bash
python ./model/1pretrainedmodelv1-MutiMolGPT.py
```

---

## 2. Multi-Modal Multi-Target Training

Script:

```bash
./model/2MM-MT-surf-drugdesignv10.py
```

Example:

```bash
python ./model/2MM-MT-surf-drugdesignv10.py
```

---

# Experimental Results

All generated molecules, docking scores, and evaluation results are stored in:

```bash
./result
```

The folder includes:

- Generated candidate molecules
- Docking and binding affinity results
- Molecular property evaluation
- Multi-target screening results
- Visualization and analysis outputs

---

# Directory Structure

```bash
.
├── data
│   ├── ZINC15
│   ├── multidata
│   │   ├── filtered_dual_target_data2.csv
│   │   ├── stru_data
│   │   └── prot_emb_data
│   ├── 1screenpretraindatafromZINC15.py
│   ├── 2pretrainedsmiles_properties.py
│   ├── 3PDB2graph.py
│   └── 4screenmultimodaldata.py
│
├── model
│   ├── 1pretrainedmodelv1-MutiMolGPT.py
│   └── 2MM-MT-surf-drugdesignv10.py
│
├── result
│
├── environment.yml
└── README.md
```

---

# Key Features

- Pocket-centric protein topology encoding
- Geometric protein structure modeling
- Multi-modal protein representation fusion
- Multi-target drug generation
- Multi-property optimization
- Large-scale molecular pretraining
- Polypharmacology-guided molecular design

---

# Citation

```bibtex
@inproceedings{liu2026geometric,
  title={Geometric Pocket-Centric Protein Encoding for Polypharmacology-Guided Multi-Target Drug Design},
  author={Liu, Haoran and others},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2026}
}
```

---

# License

This project is released for academic research purposes only.
