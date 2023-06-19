import pandas as pd
from model import TransCDR
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index

response = pd.read_csv('./data/CCLE/data_processed/to_GDSC_form2/response_n9242.csv')
# response = response[response['IC50 (uM)']!=8]
# response = response.reset_index(drop=True)
response['Label'] = np.log(response['IC50 (uM)'])
len(response['cell_lines'].unique()) # 401
len(response['Compound'].unique()) # 24
CDR = pd.read_csv('./data/GDSC/data_processed/CDR_n156813.txt',sep='\t')
drugs = np.intersect1d(response['smiles'],CDR['smiles']) # 14
drugs = pd.DataFrame(drugs)
drugs.columns = ['smiles']
response = pd.merge(response,drugs,on='smiles')


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
        'external_dataset':'CCLE'
    }
net = TransCDR(**config)
net.load_pretrained(path = './result/Final_model/regression/model.pt')
y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = net.predict(response)

response['y_pred'] = y_pred
response.to_csv('./result/CCLE_test/response_n5452.csv')
# cancer types
rmse = []
person = []
spearman = []
CI = []
cancers =  response['primary_disease'].value_counts().index.values
for i in cancers:
    df = response[response['primary_disease']==i]
    y_label = df['Label']
    y_pred = df['y_pred']
    rmse.append(np.sqrt(mean_squared_error(y_label, y_pred)))
    person.append(pearsonr(y_label, y_pred)[0])
    spearman.append(spearmanr(y_label, y_pred)[0])
    CI.append(concordance_index(y_label, y_pred))
    

res = pd.DataFrame({'rmse':rmse,'person':person,'spearman':spearman,'CI':CI})
res.index = cancers
res['n_sample'] = response['primary_disease'].value_counts().values
res = res.sort_values(by='person')
res
res.to_csv('./result/CCLE_test/res_by_cancer.csv')
# drugs
rmse = []
person = []
spearman = []
CI = []
drugs =  response['Compound'].value_counts().index.values
for i in drugs:
    df = response[response['Compound']==i]
    y_label = df['Label']
    y_pred = df['y_pred']
    rmse.append(np.sqrt(mean_squared_error(y_label, y_pred)))
    person.append(pearsonr(y_label, y_pred)[0])
    spearman.append(spearmanr(y_label, y_pred)[0])
    CI.append(concordance_index(y_label, y_pred))
    

res = pd.DataFrame({'rmse':rmse,'person':person,'spearman':spearman,'CI':CI})
res['n_sample'] = response['Compound'].value_counts().values
res.index = drugs
res = res.sort_values(by='person')
res
res.to_csv('./result/CCLE_test/res_by_drug.csv')
