AA_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY", start=1)}  # 1-indexed
GENE_MAP = {'TRBV6-1': 1, 'TRBV7-3': 2, 'TRBV5-4': 3, 'TRBJ2-1': 1, 'TRBJ2-7': 2, 'TRBJ1-2': 3}

def featurize_sequences(seq_list):
    max_len = 20
    return [[AA_MAP.get(aa, 0) for aa in seq.ljust(max_len, 'X')[:max_len]] for seq in seq_list]

def featurize_genes(genes):
    return [GENE_MAP.get(g, 0) for g in genes]
