import os 
import pandas as pd
import numpy as np
from model import TransCDR

# prepare patient drug pairs in each drug
all_types = os.listdir('./data/TCGA/omics_TCGA/processed_matrix_screening')
CDR = pd.read_csv('./data/GDSC/data_processed/CDR_n156813.txt',sep='\t')
drugs_info = CDR[['drug_id','drug_name','smiles']].drop_duplicates()

for cancer_type in all_types:
    print(cancer_type)
    patient_id = pd.read_csv('./data/TCGA/omics_TCGA/processed_matrix_screening/'+cancer_type+'/patient_id.csv',index_col=0)
    for drug in drugs_info['drug_id']:
        drug_name = drugs_info[drugs_info['drug_id']==drug]['drug_name'].values[0]
        CDPs = pd.DataFrame({'drug_id':np.tile(drug, (len(patient_id),)),'patient.arr':patient_id['patient.arr']})
        CDPs = pd.merge(CDPs,drugs_info,on='drug_id')
        CDPs['Label'] = 0
        CDPs['Label'][0:3] = 1
        config = {
            'model_type':'classification', # classification, regression
            'omics':'expr + mutation + methylation', # expr + mutation + methylation
            'input_dim_drug':300+768+1024,  # graph:300, seq:768, sequence + graph:1068
            'input_dim_rna':18451,
            'input_dim_genetic':735,
            'input_dim_mrna':20617,
            'KG':'',
            'pre_train':True,
            'fusion_type':'encoder', # concat, decoder,encoder
            'drug_encoder':'', # CNN, RNN, Transformer, GCN, NeuralFP,AttentiveFP
            'drug_model': 'sequence + graph + FP', # graph, sequence, sequence + graph FP
            'modeldir':' ',
            'seq_model':'seyonec/ChemBERTa-zinc-base-v1',
            'graph_model':'gin_supervised_masking',
            'external_dataset':'None',
            'screening':'TCGA',
            'cancer_type':cancer_type
            }
        net = TransCDR(**config)
        net.load_pretrained(path = './result/Final_model/classification/model.pt')
        auc, pr,f1,loss,y_pred,y_label = net.predict(CDPs)
        CDPs['y_pred'] = y_pred
        CDPs = CDPs.sort_values(by='y_pred',ascending = False)
        res_path = './result/TCGA_screening/'+cancer_type
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        CDPs.to_csv(res_path+'/'+drug_name+'_response.csv')

# analysis
all_response = []
all_patients = []
for cancer_type in all_types:
    cancer_response = []
    for drug in drugs_info['drug_name'].unique():
        response = pd.read_csv('./result/TCGA_screening/'+cancer_type+'/'+drug+'_response.csv')
        cancer_response.append(response)
    patient = pd.read_csv('./data/TCGA/omics_TCGA/processed_matrix_screening/'+cancer_type+'/patient_id.csv',index_col=0)
    patient['cancer_types'] = cancer_type
    cancer_response = pd.concat(cancer_response,axis=0)
    cancer_response = cancer_response.sort_values(by='y_pred',ascending=False)
    cancer_response = pd.merge(cancer_response,patient,on='patient.arr')
    cancer_response.to_csv('./data/TCGA/omics_TCGA/processed_matrix_screening/'+cancer_type+'/cancer_response.csv')
    all_response.append(cancer_response)
    all_patients.append(patient)

all_response = pd.concat(all_response,axis=0)
all_response = all_response.sort_values(by='y_pred',ascending=False)
all_response.to_csv('./result/TCGA_screening/all_response.csv')
all_patients = pd.concat(all_patients,axis=0)
all_patients.to_csv('./result/TCGA_screening/all_patients.csv')
# 针对每种药物
response_drugs = []
for drug in drugs_info['drug_name'].unique():
    print(drug)
    response_d = all_response[all_response['drug_name']==drug]
    response_top5 = response_d.iloc[0:round(len(response_d)*0.05)]
    response_tail5 = response_d.iloc[-round(len(response_d)*0.05):]
    response_top5['response'] = 'sensitive'
    response_tail5['response'] = 'resistant'
    # response_top5['cancer_types'].value_counts()
    # response_tail5['cancer_types'].value_counts()
    response_drugs.append(pd.concat([response_top5,response_tail5],axis=0))

response_drugs = pd.concat(response_drugs,axis=0)
response_drugs = response_drugs.sort_values(by='y_pred',ascending=False)
response_drugs.to_csv('./result/TCGA_screening/all_drug_response_top5_tail5.csv')
response_drugs['patient.arr'].value_counts()
response_drugs['drug_name'].value_counts()

# 提取每种药物top5%和tail5%的病人的表达矩阵
patient_id = response_drugs['patient.arr'].unique()
GE_all = []
N_sample = []
for cancer_type in all_types:
    print(cancer_type)
    GE = pd.read_csv('./data/TCGA/omics_TCGA/processed_matrix_screening/'+cancer_type+'/EG_tcga_scale.csv',index_col=0)
    samples = np.intersect1d(patient_id,GE.columns.values)
    N_sample.append(len(samples))
    GE_all.append(GE[samples])

N_sample = pd.DataFrame(N_sample)
N_sample.columns = ['N_sample']
N_sample['cancer_type'] = all_types
GE_all = pd.concat(GE_all,axis=1) # 18451 x 6684
N_sample.to_csv('./result/TCGA_screening/N_sample.csv')
GE_all.to_csv('./result/TCGA_screening/GE_all.csv')


# 选择sensitive 和resistant差异最大的药物
drug_names = response_drugs['drug_name'].unique()
sen_values = []
res_values = []
for drug in drug_names:
    print(drug)
    response_d = response_drugs[response_drugs['drug_name']==drug]
    sen_values.append(response_d[response_d['response']=='sensitive']['y_pred'].mean())
    res_values.append(response_d[response_d['response']=='resistant']['y_pred'].mean())

res = pd.DataFrame({'drug_names':drug_names,
                    'sen_values':sen_values,
                    'res_values':res_values})
res['diff'] = res['sen_values']-res['res_values']
res = res.sort_values(by='diff',ascending=False)
res.to_csv('./result/TCGA_screening/GSEA_analysis/drug_for_GSEA_analysis.csv')

# 得到每种药物对应的表达矩阵
drug = 'Dasatinib'
response_d = response_drugs[response_drugs['drug_name']==drug]
GE = GE_all[response_d['patient.arr'].values]
path = './result/TCGA_screening/GSEA_analysis/'+drug
if not os.path.exists(path):
    os.makedirs(path)

response_d['response'].to_csv(path+'/sample_info.csv')
GE.to_csv(path+'/GE.txt',sep='\t')
