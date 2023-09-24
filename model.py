# python3
# -*- coding:utf-8 -*-

"""
@author:Xiaoqiong Xia
@e-mail: 
@model: multi-omics model
@time:2021/9/15 16:33 
"""

import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
import copy
import time
import pickle
import dgl
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tqdm import tqdm
from dgllife.model import load_pretrained
from dgl.nn.pytorch.glob import AvgPooling
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
#import sys
#sys.path.append('./code/cross_att_clas_regr')
from model_helper import MultiHeadAttention,Encoder,Decoder,CNN,RNN,Transformer,GCN_encoder,NeuralFP, AttentiveFP
from drug_bert_model import LigandTokenizer, embed_ligand
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# get sequence feats with pre-trained BERT model
def get_sequence_feats(drug_df,**config):
    chem_tokenizer = LigandTokenizer(**config)
    # seyonec/PubChem10M_SMILES_BPE_450k, seyonec/ChemBERTa-zinc-base-v1
    lig_embedder = AutoModel.from_pretrained(config['seq_model'])
    lig_embedder.eval()
    embedded_drug = {}
    smiles_list = pd.Series(drug_df['smiles'].unique())
    for smiles in tqdm(smiles_list):
        embedded_drug[smiles] = embed_ligand(smiles, chem_tokenizer, lig_embedder)  
    return embedded_drug


# get graph feats with pre-trained GIN model
def get_graph_feats(drug_df,**config):
    model_drug = load_pretrained(config['graph_model'])
    readout = AvgPooling()
    model_drug.eval()
    model_drug.to(device)
    embedded_drug = {}
    smiles_list = pd.Series(drug_df['smiles'].unique())
    for smiles in tqdm(smiles_list):
        v_D = mol_to_bigraph(Chem.MolFromSmiles(smiles), add_self_loop=True,
                                                    node_featurizer=PretrainAtomFeaturizer(),
                                                    edge_featurizer=PretrainBondFeaturizer(),
                                                    canonical_atom_order=False)
        v_D = v_D.to(device)
        nfeats = [v_D.ndata.pop('atomic_number').to(device),
                v_D.ndata.pop('chirality_type').to(device)]
        efeats = [v_D.edata.pop('bond_type').to(device),
                v_D.edata.pop('bond_direction_type').to(device)]
        node_repr = model_drug(v_D, nfeats, efeats)
        v_D = readout(v_D, node_repr)
        embedded_drug[smiles] = v_D.detach().cpu().numpy().flatten()   
    return embedded_drug

# get ECFP feats
def smiles2morgan(smiles, radius = 2, nBits = 1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' + smiles + ' convert to all 0 features')
        features = np.zeros((nBits, )) 
    return features

def get_FP_feats(drug_df,**config):
    embedded_drug = {}
    smiles_list = pd.Series(drug_df['smiles'].unique())
    for smiles in tqdm(smiles_list):
        embedded_drug[smiles] = smiles2morgan(smiles, radius = 2, nBits = 1024)   
    return embedded_drug


# CNN & RNN
smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

def trans_drug(x):
    MAX_SEQ_DRUG = 100
    temp = list(x)
    temp = [i if i in smiles_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_DRUG:
        temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
    else:
        temp = temp [:MAX_SEQ_DRUG]
    return temp

def drug_2_embed(drug_df):
    from sklearn.preprocessing import OneHotEncoder
    enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
    smiles_list = pd.Series(drug_df['smiles'].unique())
    embedded_drug = {}
    for smiles in tqdm(smiles_list):
        smiles_split = trans_drug(smiles)
        embedded_drug[smiles] = enc_drug.transform(np.array(smiles_split).reshape(-1,1)).toarray().T
    return embedded_drug


# transformer
from subword_nmt.apply_bpe import BPE # Byte-Pair-Encoder
import codecs
vocab_path = './data/ESPF/drug_codes_chembl_freq_1500.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./data/ESPF/subword_units_map_chembl_freq_1500.csv')
idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

def drug2emb_encoder(x):
    max_d = 50
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d
    return i, np.asarray(input_mask)

def get_transformer_feats(drug_df):
    smiles_list = pd.Series(drug_df['smiles'].unique())
    embedded_drug = {}
    mask = {}
    for smiles in tqdm(smiles_list):
        embedded_drug[smiles],mask[smiles] = drug2emb_encoder(smiles)
    return embedded_drug,mask


# GCN and NeuralFP
def get_GNN_feats(drug_df,**config):
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer,AttentiveFPAtomFeaturizer,AttentiveFPBondFeaturizer
    from functools import partial
    if config['drug_encoder'] in ['GCN', 'NeuralFP']:
        node_featurizer = CanonicalAtomFeaturizer()
        edge_featurizer = CanonicalBondFeaturizer(self_loop = True)
        fc = partial(smiles_to_bigraph, add_self_loop=True)
    if config['drug_encoder'] == 'AttentiveFP':
        node_featurizer = AttentiveFPAtomFeaturizer()
        edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
        fc = partial(smiles_to_bigraph, add_self_loop=True)
    smiles_list = pd.Series(drug_df['smiles'].unique())
    embedded_drug = {}
    for smiles in tqdm(smiles_list): 
        embedded_drug[smiles] = fc(smiles = smiles, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer)
    return embedded_drug


class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, drug_df,**config):
        'Initialization'
        self.config = config
        self.labels = labels
        self.list_IDs = list_IDs
        self.drug_df = drug_df
        if self.config['screening'] == 'TCGA':
            cancer_type = self.config['cancer_type']
            self.rna_data = pd.read_csv('./data/TCGA/omics_TCGA/processed_matrix_screening/'+cancer_type+'/EG_tcga_scale.csv',index_col=0)
            # self.rna_data.columns = [name.split('.')[0][1:] for name in self.rna_data.columns.values]
            self.rna_data = self.rna_data.T
            self.genetic = pd.read_csv('./data/TCGA/omics_TCGA/processed_matrix_screening/'+cancer_type+'/MC_tcga_scale.csv',index_col=0)
            self.genetic = self.genetic.T
            self.mrna = pd.read_csv('./data/TCGA/omics_TCGA/processed_matrix_screening/'+cancer_type+'/DM_tcga_scale.csv',index_col=0)
            self.mrna = self.mrna.T
        else:
            if self.config['external_dataset'] == 'None':
                self.rna_data = pd.read_csv('./data/GDSC/data_processed/RNA_n18451_1018_zscore.csv',index_col=0)
                self.rna_data.columns = [name.split('.')[0][1:] for name in self.rna_data.columns.values]
                self.rna_data = self.rna_data.T
                self.genetic = pd.read_csv('./data/GDSC/data_processed/Genetic_features_n969_735.txt',sep='\t',index_col=0)
                self.mrna = pd.read_csv('./data/GDSC/data_processed/mrna_n20617_1028_zscore.csv',index_col=0)
                self.mrna = self.mrna.T
            if self.config['external_dataset'] == 'TCGA':
                self.rna_data = pd.read_csv('./data/TCGA/omics_TCGA/processed_matrix/EG_tcga_scale.csv',index_col=0)
                # self.rna_data.columns = [name.split('.')[0][1:] for name in self.rna_data.columns.values]
                self.rna_data = self.rna_data.T
                self.genetic = pd.read_csv('./data/TCGA/omics_TCGA/processed_matrix/MC_tcga.csv',index_col=0)
                self.genetic = self.genetic.T
                self.mrna = pd.read_csv('./data/TCGA/omics_TCGA/processed_matrix/DM_tcga_scale.csv',index_col=0)
                self.mrna = self.mrna.T
            if self.config['external_dataset'] == 'CCLE':
                self.rna_data = pd.read_csv('./data/CCLE/data_processed/to_GDSC_form2/EG_scale.csv',index_col=0)
                self.rna_data = self.rna_data.T
                self.genetic = pd.read_csv('./data/CCLE/data_processed/to_GDSC_form2/MC.csv',index_col=0)
                self.mrna = pd.read_csv('./data/CCLE/data_processed/to_GDSC_form2/DM_scale.csv',index_col=0)
                self.mrna = self.mrna.T
        if self.config['pre_train']:
            self.embedded_drug1 = get_sequence_feats(drug_df, **self.config)
            self.embedded_drug2 = get_graph_feats(drug_df, **self.config)
            self.embedded_drug3 = get_FP_feats(drug_df, **self.config)
        else:
            if (self.config['drug_encoder'] == 'CNN') | (self.config['drug_encoder'] == 'RNN'):
                self.embedded_drug = drug_2_embed(drug_df)
            if self.config['drug_encoder'] == 'Transformer':
                self.embedded_drug,self.mask = get_transformer_feats(drug_df)
            if self.config['drug_encoder'] in ['GCN','NeuralFP','AttentiveFP']:
                self.embedded_drug = get_GNN_feats(drug_df,**self.config)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        y = self.labels[index]
        if self.config['screening'] == 'TCGA':
            v_rna = np.array(self.rna_data.loc[self.drug_df.iloc[index]['patient.arr'],:])
            v_genetic = np.array(self.genetic.loc[self.drug_df.iloc[index]['patient.arr'],:])
            v_mrna = np.array(self.mrna.loc[self.drug_df.iloc[index]['patient.arr'],:])
        else:
            if self.config['external_dataset'] == 'None':
                v_rna = np.array(self.rna_data.loc[self.drug_df.iloc[index]['assay_name'],:])
                v_genetic = np.array(self.genetic.loc[int(self.drug_df.iloc[index]['COSMIC_ID']),:])
                v_mrna = np.array(self.mrna.loc[self.drug_df.iloc[index]['cell_type'],:])
            if self.config['external_dataset'] == 'TCGA':
                v_rna = np.array(self.rna_data.loc[self.drug_df.iloc[index]['patient.arr'],:])
                v_genetic = np.array(self.genetic.loc[self.drug_df.iloc[index]['patient.arr'],:])
                v_mrna = np.array(self.mrna.loc[self.drug_df.iloc[index]['patient.arr'],:])
            if self.config['external_dataset'] == 'CCLE':
                v_rna = np.array(self.rna_data.loc[self.drug_df.iloc[index]['DepMap_ID'],:])
                v_genetic = np.array(self.genetic.loc[self.drug_df.iloc[index]['cell_lines'],:])
                v_mrna = np.array(self.mrna.loc[self.drug_df.iloc[index]['cell_lines'],:])
        if self.config['pre_train']:
            v_d1 = self.embedded_drug1[self.drug_df.iloc[index]['smiles']]
            v_d2 = self.embedded_drug2[self.drug_df.iloc[index]['smiles']]
            v_d3 = self.embedded_drug3[self.drug_df.iloc[index]['smiles']]
            return v_d1,v_d2,v_d3, v_rna, v_genetic, v_mrna, y
        else:
            if self.config['drug_encoder'] in ['CNN','RNN','GCN','NeuralFP','AttentiveFP']:
                v_d = self.embedded_drug[self.drug_df.iloc[index]['smiles']]
                return v_d,v_rna, v_genetic, v_mrna, y
            if self.config['drug_encoder'] == 'Transformer':
                v_d = self.embedded_drug[self.drug_df.iloc[index]['smiles']]
                mask = self.mask[self.drug_df.iloc[index]['smiles']]
                return (v_d,mask),v_rna, v_genetic, v_mrna, y


class MLP(nn.Sequential):
    def __init__(self,input_dim_gene):
        self.input_dim_gene = input_dim_gene
        hidden_dim_gene = 256
        mlp_hidden_dims_gene = [1024, 512]
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [self.input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])
    def forward(self, v):
        # predict
        v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v


class Classifier(nn.Sequential):
    def __init__(self, **config):
        super(Classifier, self).__init__()
        self.config = config
        
        # pretrained models
        self.model_seq = MLP(768)
        self.model_graph = MLP(300)
        self.model_fp = MLP(1024)

        # cell line model
        self.model_rna = MLP(self.config['input_dim_rna'])
        self.model_genetic = MLP(self.config['input_dim_genetic'])
        self.model_mrna = MLP(self.config['input_dim_mrna'])

        # drug_encoder
        if self.config['drug_encoder'] == 'CNN':
            self.drug_model = CNN()
        
        if self.config['drug_encoder'] == 'RNN':
            self.drug_model = RNN()
        
        if self.config['drug_encoder'] == 'Transformer':
            self.drug_model = Transformer()
        
        if self.config['drug_encoder'] == 'GCN':
            self.drug_model = GCN_encoder()
        
        if self.config['drug_encoder'] == 'NeuralFP':
            self.drug_model = NeuralFP()
        
        if self.config['drug_encoder'] == 'AttentiveFP':
            self.drug_model = AttentiveFP()
        # fusion
        # self.fusion = MultiHeadAttention(256, 8)
        # self.fusion = Encoder(256, 256, 8, 4, 0.1, device)
        if self.config['fusion_type'] =='encoder':
            self.fusion = Encoder(256, 256, 8, 6, 0.1, device)
        
        if self.config['fusion_type'] =='decoder':
            self.fusion = Decoder(256,256, 256, 8, 6, 0.1, device)
            
        self.hidden_dims =  [1024, 1024, 512]
        if self.config['pre_train']:
            if self.config['fusion_type'] =='decoder':
                dims = [256*3] + self.hidden_dims + [1]
            else:
                n_drug = len(self.config['drug_model'].split(' + '))
                n_omics = len(self.config['omics'].split(' + '))
                n = n_drug + n_omics
                dims = [256*n] + self.hidden_dims + [1]
        else:
            dims = [256*4] + self.hidden_dims + [1]
        
        self.dropout = nn.Dropout(0.1)
        layer_size = len(self.hidden_dims) + 1
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # label
        label = v[-1]

        if self.config['pre_train']:
            if self.config['fusion_type'] in ['encoder','decoder']:
                # drug
                v_seq = v[0].to(device)
                v_seq = self.model_seq(v_seq)
                v_graph = v[1].to(device)
                v_graph = self.model_graph(v_graph)
                v_fp = v[2].to(device)
                v_fp = self.model_fp(v_fp)
                # unsquence
                v_seq = v_seq.unsqueeze(1)
                v_graph = v_graph.unsqueeze(1)
                v_fp = v_fp.unsqueeze(1)

                # cell line
                v_rna = self.model_rna(v[3])
                v_genetic = self.model_genetic(v[4])
                v_mrna = self.model_mrna(v[5])
                # unsqueeze
                v_rna = v_rna.unsqueeze(1)
                v_genetic = v_genetic.unsqueeze(1)
                v_mrna = v_mrna.unsqueeze(1)
            else:
                #  drug
                v_seq = v[0].to(device)
                v_seq = self.model_seq(v_seq)
                v_graph = v[1].to(device)
                v_graph = self.model_graph(v_graph)
                v_fp = v[2].to(device)
                v_fp = self.model_fp(v_fp)
                
                # cell line
                v_rna = self.model_rna(v[3])
                v_genetic = self.model_genetic(v[4])
                v_mrna = self.model_mrna(v[5])

            if self.config['drug_model'] == 'sequence + graph + FP':
                v_D = torch.cat((v_seq,v_graph,v_fp),1)
            if self.config['drug_model'] == 'sequence + graph':
                v_D = torch.cat((v_seq,v_graph),1)
            if self.config['drug_model'] == 'sequence + FP':
                v_D = torch.cat((v_seq,v_fp),1)
            if self.config['drug_model'] == 'graph + FP':
                v_D = torch.cat((v_graph,v_fp),1)
            if self.config['drug_model'] == 'sequence':
                v_D = v_seq
            if self.config['drug_model'] == 'graph':
                v_D = v_graph
            if self.config['drug_model'] == 'FP':
                v_D = v_fp
            if self.config['omics'] == 'expr + mutation + methylation':
                v_cell = torch.cat((v_rna, v_genetic, v_mrna), 1)
            if self.config['omics'] == 'expr + mutation':
                v_cell = torch.cat((v_rna, v_genetic), 1)
            if self.config['omics'] == 'expr + methylation':
                v_cell = torch.cat((v_rna, v_mrna), 1)
            if self.config['omics'] == 'mutation + methylation':
                v_cell = torch.cat((v_genetic, v_mrna), 1)
            if self.config['omics'] == 'expr':
                v_cell = v_rna
            if self.config['omics'] == 'mutation':
                v_cell = v_genetic
            if self.config['omics'] == 'methylation':
                v_cell = v_mrna
        else:
            # cell line
            v_rna = self.model_rna(v[1])
            v_genetic = self.model_genetic(v[2])
            v_mrna = self.model_mrna(v[3])

            # unsqueeze
            v_rna = v_rna.unsqueeze(1)
            v_genetic = v_genetic.unsqueeze(1)
            v_mrna = v_mrna.unsqueeze(1)

            v_cell = torch.cat((v_rna, v_genetic, v_mrna), 1)

            if self.config['drug_encoder'] in ['CNN','RNN','GCN','NeuralFP','AttentiveFP']:
                # drug 
                v_D = v[0].to(device)
                v_D = self.drug_model(v_D)
                v_D = v_D.unsqueeze(1)
            
            if self.config['drug_encoder'] == 'Transformer':
                v_D = v[0]
                v_D = v_D[0].to(device)
                mask = v_D[1].to(device)
                v_D = self.drug_model(v_D,mask)
                v_D = v_D.unsqueeze(1)
        
        if self.config['fusion_type'] =='decoder':
            # v_f = self.fusion(v_cell,v_D,None,None)
            v_f = self.fusion(v_D,v_cell,None,None)
            v_f = v_f.view(-1, v_f.shape[1] * v_f.shape[2])

        if self.config['fusion_type'] =='encoder':
            v_f = torch.cat((v_D, v_cell), 1) 
            v_f = self.fusion(v_f,None)
            # v_f = v_f[:,0]
            v_f = v_f.view(-1, v_f.shape[1] * v_f.shape[2])
        if self.config['fusion_type'] =='concat':
            v_f = torch.cat((v_D, v_cell), 1)

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f, label


def dgl_collate_func(x):
    d, v1,v2,v3, y = zip(*x)
    import dgl
    d = dgl.batch(d)
    return d, torch.tensor(v1), torch.tensor(v2),torch.tensor(v3), torch.tensor(y)


class TransCDR:
    def __init__(self,**config):
        #model_drug = transformer()
        self.config = config
        self.model = Classifier(**self.config)
        self.modeldir = config['modeldir']
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")

    def test(self,datagenerator,model):
        y_label = []
        y_pred = []
        model.eval()
        for i, v in enumerate(datagenerator):
            score,label = model(v)
            if self.config['model_type'] == 'regression':
                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1)
                loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
                logits = torch.squeeze(score).detach().cpu().numpy()
            
            if self.config['model_type'] == 'classification':
                m = torch.nn.Sigmoid()
                logits = torch.squeeze(m(score)).detach().cpu().numpy()

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

        model.train()

        if self.config['model_type'] == 'regression':
            return y_label, y_pred, \
                mean_squared_error(y_label, y_pred), \
                np.sqrt(mean_squared_error(y_label, y_pred)), \
                pearsonr(y_label, y_pred)[0], \
                pearsonr(y_label, y_pred)[1], \
                spearmanr(y_label, y_pred)[0], \
                spearmanr(y_label, y_pred)[1], \
                concordance_index(y_label, y_pred), \
                loss
        if self.config['model_type'] == 'classification':
            return roc_auc_score(y_label, y_pred), \
                average_precision_score(y_label, y_pred), \
                f1_score(y_label, outputs), \
                log_loss(y_label, outputs), \
                y_pred,y_label

    def train(self, train_drug, test_drug, val_drug):
        lr = self.config['lr']
        decay = self.config['decay']
        BATCH_SIZE = self.config['BATCH_SIZE']
        train_epoch = self.config['train_epoch']
        self.model = self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 5])
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=decay)
        loss_history = []
        
        print('******** data preparing ********')
        params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': False,
                }
        if self.config['drug_encoder'] in ['GCN','NeuralFP','AttentiveFP']:
            params['collate_fn'] = dgl_collate_func

        training_generator = data.DataLoader(data_process_loader(
            train_drug.index.values, train_drug.Label.values, train_drug,**self.config), **params)
        if test_drug is not None:
            testing_generator = data.DataLoader(data_process_loader(
                test_drug.index.values, test_drug.Label.values, test_drug,**self.config), **params)
        if val_drug is not None:
            validation_generator = data.DataLoader(data_process_loader(
                val_drug.index.values, val_drug.Label.values, val_drug,**self.config), **params)

        valid_metric_record = []
        if self.config['model_type'] == 'regression':
            max_MSE = 10000
            valid_metric_header = ['# epoch',"MSE", 'RMSE',
                                "Pearson Correlation", "with p-value",
                                'Spearman Correlation',"with p-value2",
                                "Concordance Index"]

        if self.config['model_type'] == 'classification':
            max_auc = 0
            valid_metric_header = ['# epoch',"AUROC", "AUPRC", "F1"]
        
        model_max = copy.deepcopy(self.model)
        table = PrettyTable(valid_metric_header)
        float2str = lambda x: '%0.4f' % x
        
        print('******** Go for Training ********')
        writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_MLP')
        t_start = time.time()
        iteration_loss = 0

        for epo in range(train_epoch):
            for i, v in enumerate(training_generator):
                score, label = self.model(v)
                label = Variable(torch.from_numpy(np.array(label))).float().to(self.device)
                
                if self.config['model_type'] == 'regression':
                    loss_fct = torch.nn.MSELoss()
                    n = torch.squeeze(score, 1).float()
                    loss = loss_fct(n, label)
                
                if self.config['model_type'] == 'classification':
                    loss_fct = torch.nn.BCELoss()
                    m = torch.nn.Sigmoid()
                    n = torch.squeeze(m(score), 1)
                    loss = loss_fct(n, label)

                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()
                if (i % 100 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                          ' iteration ' + str(i) + \
                          ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                          ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")
            if val_drug is not None:
                with torch.set_grad_enabled(False):
                    if self.config['model_type'] == 'regression':
                        ### regression: MSE, Pearson Correlation, with p-value, Concordance Index
                        y_true,y_pred, mse, rmse, \
                        person, p_val, \
                        spearman, s_p_val, CI,\
                        loss_val = self.test(validation_generator, self.model)
                        lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                        s_p_val, CI]))
                        valid_metric_record.append(lst)
                        print('Validation at Epoch ' + str(epo + 1) +
                                ' with loss:' + str(loss_val.item())[:7] +
                                ', MSE: ' + str(mse)[:7] +
                                ' , Pearson Correlation: ' + str(person)[:7] +
                                ' with p-value: ' + str(p_val)[:7] +
                                ' Spearman Correlation: ' + str(spearman)[:7] +
                                ' with p_value: ' + str(s_p_val)[:7] +
                                ' , Concordance Index: ' + str(CI)[:7])
                        writer.add_scalar("valid/mse", mse, epo)
                        writer.add_scalar('valida/rmse', rmse, epo)
                        writer.add_scalar("valid/pearson_correlation", person, epo)
                        writer.add_scalar("valid/concordance_index", CI, epo)
                        writer.add_scalar("valid/Spearman", spearman, epo)
                        writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
                        if mse < max_MSE:
                            model_max = copy.deepcopy(self.model)
                            max_MSE = mse
                            es = 0
                        else:
                            es += 1
                            print("Counter {} of 5".format(es))
                            if es > 4:
                                print("Early stopping with best_mse: ", str(max_MSE)[:7], "and mse for this epoch: ", str(mse)[:7], "...")
                                break
                    
                    if self.config['model_type'] == 'classification':
                        auc, auprc, f1, loss, logits,_ = self.test(validation_generator, self.model)
                        lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1]))
                        valid_metric_record.append(lst)
                        print('Validation at Epoch '+ str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
                                ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
                                str(loss)[:7])
                        if auc > max_auc:
                            model_max = copy.deepcopy(self.model)
                            max_auc = auc
                            es = 0
                        else:
                            es += 1
                            print("Counter {} of 5".format(es))
                            if es > 4:
                                print("Early stopping with best_auc: ", str(max_auc)[:7], "and AUC for this epoch: ", str(auc)[:7], "...")
                                break
                table.add_row(lst)
            else:
                model_max = copy.deepcopy(self.model)
        # load early stopped model
        self.model = model_max
        
        if val_drug is not None:
            prettytable_file = os.path.join(self.modeldir, "valid_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(table.get_string())
        
        pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        with open(pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)
        
        print('******** Go for Testing ********')
        if test_drug is not None:
            if self.config['model_type'] == 'regression':
                y_true,y_pred, mse, rmse, \
                person, p_val, \
                spearman, s_p_val, CI,\
                loss_test = self.test(testing_generator, model_max)
                test_table = PrettyTable(["MSE","RMSE", "Pearson Correlation", "p-value", "spearman","s_p-value","Concordance Index"])
                test_table.add_row(list(map(float2str, [mse,rmse, person, p_val,spearman, s_p_val, CI])))
                print('Testing MSE: ' + str(mse) 
                    + ' , Pearson Correlation: ' + str(person) + ' with p-value: ' + str(f"{p_val:.2E}") 
                    + ' , Spearman Correlation: ' + str(spearman) + ' with s_p-value: ' + str(f"{s_p_val:.2E}")
                    + ' , Concordance Index: '+str(CI))
            
            if self.config['model_type'] == 'classification':
                auc, auprc, f1, loss, y_pred,_ = self.test(testing_generator, model_max)
                test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
                test_table.add_row(list(map(float2str, [auc, auprc, f1])))
                print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
                    ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
                    str(loss)[:7])
            
            np.save(os.path.join(self.modeldir, "logits.npy"), np.array(y_pred))
            # att.to_csv(os.path.join(self.modeldir, "attention.csv"))
            prettytable_file = os.path.join(self.modeldir, "test_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(test_table.get_string())

        print('******** plot learning curve ********')
        fontsize = 16
        iter_num = list(range(1,len(loss_history)+1))
        plt.figure(dpi=600)
        plt.plot(iter_num, loss_history, 'bo-')
        plt.xlabel('iteration', fontsize = fontsize)
        plt.ylabel('loss value', fontsize = fontsize)
        plt.savefig(os.path.join(self.modeldir, "loss_curve.png"))
        print('******** Training Finished ********')
        writer.flush()
        writer.close()

        tf = open(os.path.join(self.modeldir, "config.json"), "w")
        json.dump(self.config,tf)
        tf.close()
                

    def predict(self, drug_data):
        print('******** predicting... ********')
        config = self.config
        self.model.to(self.device)
        info = data_process_loader(drug_data.index.values,
                                   drug_data.Label.values,
                                   drug_data, **config)
        params = {'batch_size': 16,
                  'shuffle': False,
                  'num_workers': 8,
                  'drop_last': False,
                  'sampler': SequentialSampler(info)}
        generator = data.DataLoader(info, **params)
        
        if self.config['model_type'] == 'regression':
            y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val = \
                self.test(generator, self.model)
            return y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI
        
        if self.config['model_type'] == 'classification':
            auc, pr,f1,loss,y_pred,y_label = self.test(generator, self.model)
            return auc, pr,f1,loss,y_pred,y_label

    def save_model(self):
        torch.save(self.model.state_dict(), self.modeldir + '/model.pt')

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)








