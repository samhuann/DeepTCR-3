# functions/io.py

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict

from .featurization import featurize_sequences, featurize_genes
from .utils import save_pkl, load_pkl

class DeepTCRDataset(Dataset):
    def __init__(self, data: List[Dict], mode: str = 'sequence', use_cuda: bool = False):
        self.data = data
        self.mode = mode
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.mode == 'sequence':
            return {
                'seq': torch.tensor(item['seq'], dtype=torch.long, device=self.device),
                'v_gene': torch.tensor(item['v_gene'], dtype=torch.long, device=self.device),
                'j_gene': torch.tensor(item['j_gene'], dtype=torch.long, device=self.device),
                'label': torch.tensor(item['label'], dtype=torch.float, device=self.device),
                'weight': torch.tensor(item.get('weight', 1.0), dtype=torch.float, device=self.device)
            }
        elif self.mode == 'repertoire':
            return {
                'seqs': torch.tensor(item['seqs'], dtype=torch.long, device=self.device),
                'v_genes': torch.tensor(item['v_genes'], dtype=torch.long, device=self.device),
                'j_genes': torch.tensor(item['j_genes'], dtype=torch.long, device=self.device),
                'label': torch.tensor(item['label'], dtype=torch.float, device=self.device)
            }
        else:
            raise ValueError("Unknown mode: should be 'sequence' or 'repertoire'")


def load_tcr_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep='\t' if filepath.endswith('.tsv') else ',')
    df = df[df['sequence'].str.len() <= 40]
    df = df[df['sequence'].str.isalpha()]
    df = df[~df['sequence'].str.contains('[*X]')]
    return df


def preprocess_data(df: pd.DataFrame, label_col: str = 'label',
                    per_repertoire: bool = False) -> List[Dict]:
    data = []
    if per_repertoire:
        grouped = df.groupby('sample_id')
        for rep_id, group in grouped:
            entry = {
                'seqs': featurize_sequences(group['sequence'].tolist()),
                'v_genes': featurize_genes(group['v_gene'].tolist()),
                'j_genes': featurize_genes(group['j_gene'].tolist()),
                'label': group[label_col].iloc[0]
            }
            data.append(entry)
    else:
        for _, row in df.iterrows():
            entry = {
                'seq': featurize_sequences([row['sequence']])[0],
                'v_gene': featurize_genes([row['v_gene']])[0],
                'j_gene': featurize_genes([row['j_gene']])[0],
                'label': row[label_col],
                'weight': row.get('weight', 1.0)
            }
            data.append(entry)
    return data


def get_data(file_path: str, cache_path: Optional[str] = None, 
             per_repertoire: bool = False, label_col: str = 'label', use_cuda: bool = False) -> DeepTCRDataset:

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return load_pkl(cache_path)

    print(f"Loading TCR data from {file_path}")
    df = load_tcr_csv(file_path)
    data = preprocess_data(df, label_col=label_col, per_repertoire=per_repertoire)
    dataset = DeepTCRDataset(data, mode='repertoire' if per_repertoire else 'sequence', use_cuda=use_cuda)

    if cache_path:
        save_pkl(dataset, cache_path)

    return dataset
