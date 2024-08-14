
from pathlib import Path
from typing import List

import pandas as pd
import torch


def get_chemical_representation(
    smiles: List[str],
    embedding_model: str,
    data_dir=None,
    device="cuda",
):
    """
    Given a list of SMILES strings, returns the embeddings produced by the embedding model.
    The embeddings are loaded from disk without ever running the embedding model.

    :return: torch.nn.Embedding, shape [len(smiles), dim_embedding]. Embeddings are ordered as in `smiles`-list.
    """
    df = pd.read_parquet(data_dir)

    if df is not None:
        emb = torch.tensor(df.loc[smiles].values, dtype=torch.float32, device=device)
        assert emb.shape[0] == len(smiles)
    else:
        assert embedding_model == "zeros"
        emb = torch.zeros((len(smiles), 256))
    return torch.nn.Embedding.from_pretrained(emb, freeze=True)

def get_celltype_embeddings(
        celltype_emb_dict: dict,
        unique_celltype_list,
        device='cuda',
):
    df = pd.DataFrame(celltype_emb_dict).T
    unique_celltype_list = list(unique_celltype_list)
    emb = torch.tensor(df.loc[unique_celltype_list].values, dtype=torch.float32, device=device)
    assert emb.shape[0] == len(unique_celltype_list)
    cov_embs = torch.nn.Embedding.from_pretrained(emb, freeze=False)

    return cov_embs
