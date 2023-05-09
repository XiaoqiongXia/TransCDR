# TransCDR: a deep learning model for enhancing the generalizability of cancer drug response prediction through transfer learning and multimodal data fusion
-----------------------------------------------------------------
Code by **Xiaoqiong Xia** at Fudan University.

## 1. Introduction
**TransCDR** is a Python implementation of the a deep learning model for enhancing the generalizability of cancer drug response prediction through transfer learning and multimodal data fusion. 

**TransCDR** achieves state-of-the-art results of predicting CDRs in various scenarios. More importantly, **TransCDR** is shown to be effective in the external dataset: CCLE and TCGA. In summary, **TransCDR** could be a powerful tool for cancer drug response prediction and has promising prospects in precision medicine.

## 2. TransCDR
![alt text](docs/fig1-update.png "TransCDR")
Figure 1: The overall architecture of **TransCDR**.

## 3. Installation
**TransCDR** depends on the following packages, you must have them installed before using **TransCDR**.
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch  
conda install -c dglteam/label/cu102 dgl  
conda install -c rdkit rdkit  
pip install dgllife  
pip install matplotlib  
pip install seaborn  
pip install lifelines  
pip install prettytable  
pip install pubchempy  
pip install fitlog  
## 4. Usage
### 4.1. Data
All datasets used in the project are located at https://zenodo.org/deposit/new. You shoud download and unzip the data.7z and put it in the current directory.
The data split script for TransCDR is located at folder script/
cd script
$ bash Step1_data_split.sh
Arguments in this scripts:
--model_type: classification or regression.
--scenarios: warm start, cold drug, cold scaffold, cold cell, cold cell cluster ,cold cell & scaffold.
--n_clusters: if cold cell cluster, you should set the number of cluster.
--n_sampling: if classification, you should set the sampling ratio.
--result_folder: path for CV10 data.

### 4.2. CV10 for TransCDR
The CV10 script for TransCDR is located at folder script/
$ bash Step2_TransCDR_CV10.sh
Arguments in this scripts:
--model_type: classification or regression.
--data_path: the data path of CV10 data.
--omics: expr, mutation, methylation, expr + mutation, expr + methylation, mutation + methylation, expr + mutation + methylation.
--input_dim_drug: seq_model: 768, graph_model: 30.
--lr: Learning rate.
--BATCH_SIZE: Batch size.
--train_epoch: Number of epoch.
--pre_train: True, False.
--screening: None: traning TransCDR using GDSC dataset, TCGA: screening TCGA dataset.
--fusion_type: concat, decoder,encoder.
--drug_encoder: None, CNN, RNN, Transformer, GCN, NeuralFP,AttentiveFP.
--drug_model: sequence, graph
--modeldir: the dir of training results
--seq_model: 'seyonec/ChemBERTa-zinc-base-v1, seyonec/PubChem10M_SMILES_BPE_450k, seyonec/ChemBERTa_zinc250k_v2_40k, seyonec/SMILES_tokenized_PubChem_shard00_160k, seyonec/PubChem10M_SMILES_BPE_180k, seyonec/PubChem10M_SMILES_BPE_396_250, seyonec/ChemBERTA_PubChem1M_shard00_155k, seyonec/BPE_SELFIES_PubChem_shard00_50k, seyonec/BPE_SELFIES_PubChem_shard00_160k'
--graph_model: 'gin_supervised_contextpred, gin_supervised_infomax, gin_supervised_edgepred, gin_supervised_masking'
--external_dataset: None:traning TransCDR using GDSC dataset; CCLE, TCGA: test on external_dataset
