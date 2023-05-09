# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:38:32 2022

@author:Xiaoqiong Xia
@e-mail: 
@model: multi-omics model
@time:2021/9/15 16:33 
"""

import pandas as pd
from DataEncoding import DataEncoding
from model import TransCDR
import argparse

parser = argparse.ArgumentParser(description='CV10 of TransCDR')
parser.add_argument('--model_type', type=str, required=True, help='classification or regression')
parser.add_argument('--data_path', type=str, required=True, help='the data path of CV10 data')
parser.add_argument('--omics', type=str, required=True, help='expr, mutation, methylation, expr + mutation, expr + methylation, mutation + methylation, expr + mutation + methylation')
parser.add_argument('--input_dim_drug', type=int, required=True, help='seq_model: 768, graph_model: 300')
parser.add_argument('--lr', type=float, required=True, help='Learning rate')
parser.add_argument('--BATCH_SIZE', type=int, required=True, help='Batch size test set')
parser.add_argument('--train_epoch', type=int, required=True, help='Number of epoch')
parser.add_argument('--pre_train', type=str, required=True, help='True, False')
parser.add_argument('--screening', type=str, required=True, help='None: traning TransCDR using GDSC dataset, TCGA: screening TCGA dataset')
parser.add_argument('--fusion_type', type=str, required=True, help='concat, decoder,encoder')
parser.add_argument('--drug_encoder', type=str, required=True, help='None, CNN, RNN, Transformer, GCN, NeuralFP,AttentiveFP')
parser.add_argument('--drug_model', type=str, required=True, help='sequence, graph')
parser.add_argument('--modeldir', type=str, required=True, help='the dir of training results')
parser.add_argument('--seq_model', type=str, required=True, help='seyonec/ChemBERTa-zinc-base-v1, seyonec/PubChem10M_SMILES_BPE_450k, seyonec/ChemBERTa_zinc250k_v2_40k, seyonec/SMILES_tokenized_PubChem_shard00_160k, seyonec/PubChem10M_SMILES_BPE_180k, seyonec/PubChem10M_SMILES_BPE_396_250, seyonec/ChemBERTA_PubChem1M_shard00_155k, seyonec/BPE_SELFIES_PubChem_shard00_50k, seyonec/BPE_SELFIES_PubChem_shard00_160k')
parser.add_argument('--graph_model', type=str, required=True, help='gin_supervised_contextpred, gin_supervised_infomax, gin_supervised_edgepred, gin_supervised_masking')
parser.add_argument('--external_dataset', type=str, required=True, help='None:traning TransCDR using GDSC dataset; CCLE, TCGA: test on external_dataset')
args = parser.parse_args()



for i in range(1,11):
    train = pd.read_csv(args.data_path + '/fold'+str(i)+'/train.txt',sep='\t')
    test = pd.read_csv(args.data_path + '/fold'+str(i)+'/test.txt',sep='\t')
    val = pd.read_csv(args.data_path + '/fold'+str(i)+'/val.txt',sep='\t')
    config = {
        'model_type': args.model_type, # classification, regression
        'omics': args.omics, # expr + mutation + methylation
        'input_dim_drug': args.input_dim_drug,  # graph:300, seq:768, sequence + graph:1068
        'input_dim_rna':18451,
        'input_dim_genetic':735,
        'input_dim_mrna':20617,
        'KG':'', # no knowledge graph
        'lr':args.lr, # learning rate
        'decay':0,
        'BATCH_SIZE':args.BATCH_SIZE, # batch size
        'train_epoch':args.train_epoch, # training epoch
        'pre_train':args.pre_train,  # using transfer learning
        'screening':args.screening, # 'None': traning TransCDR using GDSC dataset; 'TCGA': screening TCGA dataset
        'fusion_type':args.fusion_type, # concat, decoder,encoder
        'drug_encoder':args.drug_encoder, # CNN, RNN, Transformer, GCN, NeuralFP,AttentiveFP
        'drug_model': args.drug_model, # graph, sequence, sequence + graph FP
        'modeldir':args.modeldir + '/fold'+str(i),
        'seq_model':args.seq_model, # the pre-trained smiles model
        'graph_model':args.graph_model,  # the pre-trained graph model
        'external_dataset':args.external_dataset # 'None':traning TransCDR using GDSC dataset; 'CCLE', 'TCGA': test on external_dataset
    }
    data_encoding = DataEncoding(**config)
    train_set, test_set, val_set = data_encoding.encode(train, test, val)
    net = TransCDR(**config)
    net.train(train_drug=train_set,
            test_drug=test_set,
            val_drug=val_set)
    net.save_model()


