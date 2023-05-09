import os 
import pandas as pd
import numpy as np
from model import TransCDR

# given a drug, predicting CDRs of the drug on all cell lines

def CDR_prediction(drug):
    CDR = pd.read_csv('./data/GDSC/data_processed/CDR_n156813.txt',sep='\t')
    response = CDR.drop_duplicates(subset=['COSMIC_ID','cell_type','assay_name'])
    response = response.reset_index(drop=True)
    response['smiles'] = drug
    response['Label'] = np.random.rand(855)  
    config = {
        'model_type':'regression', # classification, regression
        'omics':'expr + mutation + methylation', # expr + mutation + methylation
        'input_dim_drug':300+768+1024,  # graph:300, seq:768, sequence + graph:1068
        'input_dim_rna':18451,
        'input_dim_genetic':735,
        'input_dim_mrna':20617,
        'KG':'',
        'pre_train':True,
        'screening':'None',
        'fusion_type':'encoder', # concat, decoder,encoder
        'drug_encoder':'', # CNN, RNN, Transformer, GCN, NeuralFP,AttentiveFP
        'drug_model': 'sequence + graph + FP', # graph, sequence, sequence + graph FP
        'modeldir':' ',
        'seq_model':'seyonec/ChemBERTa-zinc-base-v1',
        'graph_model':'gin_supervised_masking',
        'external_dataset':'None'
    }
    net = TransCDR(**config)
    net.load_pretrained(path = './result/Final_model/regression/model.pt')
    _, y_pred, _, _, _, _, _, _, _ = net.predict(response)
    response['y_pred'] = y_pred
    response = response.sort_values(by='y_pred',ascending=True)
    response.to_csv('./result/prediction/CDR.csv')

if __name__ == '__main__':
    drug = '[H][C@@]12C[C@@H](C)[C@H](C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'
    CDR_prediction(drug)