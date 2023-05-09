import pandas as pd
import sklearn 
from model import TransCDR
import numpy as np


#get all response
response = pd.read_csv('./data/TCGA/response_n1440.csv')
response = response.drop_duplicates(subset=['patient.arr','drug.name','response']) # 1393

# get the same drugs in GDSC (cold-cell)
CDR = pd.read_csv('./data/GDSC/data_processed/CDR_bi_n154603.txt',sep='\t')
drugs = pd.DataFrame(CDR['drug_name'].unique()) 
drugs.columns = ['drug.name']
len(np.intersect1d(response['drug.name'],CDR['drug_name'])) #22
response = pd.merge(response,drugs,on='drug.name') #834
 
response['Label'] = 0
response['Label'][(response['response'] =='Complete Response') | (response['response'] =='Partial Response')]=1
response['Label'].value_counts()
response['cancers'].value_counts()
response = response.reset_index(drop=True)
config = {
        'model_type':'classification', # classification, regression
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
        'modeldir':'./result/Final_model',
        'seq_model':'seyonec/ChemBERTa-zinc-base-v1',
        'graph_model':'gin_supervised_masking',
        'external_dataset':'TCGA'
    }
net = TransCDR(**config)
net.load_pretrained(path = './result/Final_model/classification/model.pt')
auc, pr,f1,loss,y_pred,y_label = net.predict(response)
auc
pr
response['y_pred'] = y_pred

# cancer types
# AUROC = []
# AUPR = []
# label_0 = []
# label_1 = []
# for i in response['cancers'].value_counts().index[0:-1]:
#     df = response[response['cancers']==i]
#     AUROC.append(sklearn.metrics.roc_auc_score(df['Label'],df['y_pred']))
#     AUPR.append(sklearn.metrics.average_precision_score(df['Label'],df['y_pred']))
#     label_0.append(df['Label'].value_counts()[0])
#     label_1.append(df['Label'].value_counts()[1])

# res = pd.DataFrame({'AUROC':AUROC,'AUPR':AUPR,'label_0':label_0,'label_1':label_1})
# res.index = response['cancers'].value_counts().index[0:17]
# res = res.sort_values(by='AUPR')
# res
# res.to_csv('./result/TCGA_test/res_by_cancer.csv')
# drugs
# AUROC = []
# AUPR = []
# n_sample = []
# label_0 = []
# label_1 = []

# for i in response['drug.name'].value_counts().index:
#     df = response[response['drug.name']==i]
#     AUROC.append(sklearn.metrics.roc_auc_score(df['Label'],df['y_pred']))
#     AUPR.append(sklearn.metrics.average_precision_score(df['Label'],df['y_pred']))
#     n_sample.append(len(df))
#     label_0.append(df['Label'].value_counts()[0])
#     label_1.append(df['Label'].value_counts()[1])

# res = pd.DataFrame({'AUROC':AUROC,'AUPR':AUPR,'n_sample':n_sample,'label_0':label_0,'label_1':label_1})
# res.index = response['drug.name'].value_counts().index[0:4]
# res = res.sort_values(by='AUPR')
# res
# res.to_csv('./result/TCGA_test/res_by_drug.csv')

