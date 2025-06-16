# functions/featurization.py

# Matching DeepTCR-style featurization

import numpy as np
import torch

# Amino acid vocab (DeepTCR-style)
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
           'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_TO_INDEX = {aa: i+1 for i, aa in enumerate(AA_LIST)}  # 0 is padding

# Gene vocab dictionaries (examples)
GENE_VOCAB = {}
GENE_OFFSET = 1  # 0 is padding


def featurize_sequences(seq_list, max_length=40):
    encoded = []
    for seq in seq_list:
        seq_encoded = [AA_TO_INDEX.get(aa, 0) for aa in seq[:max_length]]
        # Right pad with 0s
        if len(seq_encoded) < max_length:
            seq_encoded += [0] * (max_length - len(seq_encoded))
        encoded.append(seq_encoded)
    return encoded


def featurize_genes(genes):
    global GENE_VOCAB
    encoded = []
    for g in genes:
        if g not in GENE_VOCAB:
            GENE_VOCAB[g] = len(GENE_VOCAB) + GENE_OFFSET
        encoded.append(GENE_VOCAB[g])
    return encoded
