# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:28:23 2022

@author: toxicant
"""
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import numpy as np
import torch
from tqdm import tqdm

class LigandTokenizer:
    def __init__(self,**config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['seq_model'])

    def tokenize(self, smiles, only_tokens=False):
        tokens = self.tokenizer(smiles)
        if not (only_tokens):
            return tokens
        return list(np.array(tokens['input_ids'], dtype="int"))

def embed_ligand(smiles, chem_tokenizer, lig_embedder):
    if len(smiles) > 512:
        smiles = smiles[:512]
    tokens = chem_tokenizer.tokenize(smiles, False)
    input_ligand = torch.LongTensor([tokens['input_ids']])
    try:
        output = lig_embedder(input_ligand, return_dict=True)
    except:
        print(smiles)
        raise AssertionError
    return torch.mean(output.last_hidden_state[0], axis=0).cpu().detach().numpy()

# embedded_drug = {}
# chem_tokenizer = LigandTokenizer()
# lig_embedder = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
# lig_embedder.eval()
# print('Getting embedding SMILES feature')

# for smiles in tqdm(train_set['smiles']):
#     embedded_drug[smiles] = embed_ligand(smiles, chem_tokenizer, lig_embedder)