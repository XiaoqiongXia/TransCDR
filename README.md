# TransCDR: a deep learning model for enhancing the generalizability of cancer drug response prediction through transfer learning and multimodal data fusion
-----------------------------------------------------------------

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
All datasets used in the project are located at https://zenodo.org/deposit/new. You shoud download and unzip the result.7z and put it in the current directory.  
The data split script for TransCDR is located at folder script/  

cd script  
$ bash Step1_data_split.sh  
 
### 4.2. CV10 for TransCDR  
The CV10 script for TransCDR is located at folder script/  

$ bash Step2_TransCDR_CV10.sh 

get the CV10 results  
$ bash Step3_CV10_result.sh  

### 4.3. Training the final TransCDR  
$ bash Step4_Train_final_model.sh  

### 4.4. test the trained TransCDR on TCGA and CCLE  
The pre-trained TransCDR models are located at https://zenodo.org/deposit/new. You shoud download and unzip the data.7z and put it in the current directory.    

$ bash Step5.1_test_on_TCGA_data.sh  
$ bash Step5.2_test_on_CCLE_data.sh  

### 4.5. screening drugs for TCGA patients
bash Step5.3_screening_drugs_for_TCGA_patients.sh  

### 4.6. predicting CDRs of a drug  
$ python Step6_CDR_prediction.py  

## 5. Contact  

**Xiaoqiong Xia** < 19111510052@fudan.edu.cn >  

Department of the Institutes of Biomedical Sciences at Fudan University.   


