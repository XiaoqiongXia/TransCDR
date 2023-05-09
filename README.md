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
All datasets used in the project are located at https://zenodo.org/deposit/new. You shoud download and unzip the data.7z and put it in current directory.
You can run .py to train TransCDR.
### 4.2. Training TransCDR
You can run train_model.py to train TransCDR.
