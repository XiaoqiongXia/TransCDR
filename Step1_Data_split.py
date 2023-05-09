import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
#import sys
#sys.path.append('./code/cross_att_clas_regr')
import argparse

parser = argparse.ArgumentParser(description='data segmentation strategies')
parser.add_argument('--model_type', type=str, required=True, help='classification or regression')
parser.add_argument('--scenarios', type=str, required=True, help='warm start, cold drug, cold scaffold, cold cell, cold cell cluster ,cold cell & scaffold')
parser.add_argument('--n_clusters', type=int, required=True, help='the number of cell cluster: 10, 50, 100, 200')
parser.add_argument('--n_sampling', type=int, required=True, help='the number of sampling ratio (neg/pos): 1, 2, 5')
parser.add_argument('--result_folder', type=str, required=True, help='The save path of CV10 data')
args = parser.parse_args()
if args.model_type == 'regression':
    CDR = pd.read_csv('./data/GDSC/data_processed/CDR_n156813.txt',sep='\t',index_col=0)
    CDR = shuffle(CDR,random_state=2022) 
    
if args.model_type == 'classification':
    from random import sample
    CDR = pd.read_csv('./data/GDSC/data_processed/CDR_bi_n154603.txt',sep='\t',index_col=0)
    # dowm sampling
    CDR['Label'].value_counts()
    # 0    136460
    # 1     18143
    CDR = shuffle(CDR,random_state=2022) 
    CDR_1 = CDR[CDR['Label']==1]
    CDR_0 = CDR[CDR['Label']==0]
    CDR_0 = CDR_0.sample(n=args.n_sampling*len(CDR_1))
    CDR = pd.concat([CDR_0,CDR_1])
    CDR = shuffle(CDR,random_state=2022) 

if args.scenarios == 'warm start':
    kf = KFold(n_splits=10,random_state=2022,shuffle=True)
    i=0
    for train_index, test_index in kf.split(CDR):
            i=i+1
            result_folder = args.result_folder +'/CV10/fold'+str(i)                
            train_val, test_set = CDR.iloc[train_index], CDR.iloc[test_index]
            train_set, val_set = train_test_split(train_val, test_size = 1/9, random_state = 0)
            if os.path.exists(result_folder): 
                train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
                test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
                val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)
            else:
                os.makedirs(result_folder)
                train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
                test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
                val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)

if args.scenarios == 'cold drug':
    drugs = pd.DataFrame({'drug_id':CDR['drug_id'].unique()}) 
    kf = KFold(n_splits=10,random_state=2022,shuffle=True)
    i=0
    for train_index, test_index in kf.split(drugs):
            i=i+1
            result_folder = args.result_folder +'/CV10_cold_drug/fold'+str(i)                
            train_val, test_set = drugs.iloc[train_index], drugs.iloc[test_index]
            train_set, val_set = train_test_split(train_val, test_size = 1/9, random_state = 0)
            train_set = pd.merge(train_set,CDR,on='drug_id')
            val_set = pd.merge(val_set,CDR,on='drug_id')
            test_set = pd.merge(test_set,CDR,on='drug_id')
            if os.path.exists(result_folder): 
                train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
                test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
                val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)
            else:
                os.makedirs(result_folder)
                train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
                test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
                val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)

if args.scenarios == 'cold cell':
    cells = pd.DataFrame({'COSMIC_ID':CDR['COSMIC_ID'].unique()}) 
    kf = KFold(n_splits=10,random_state=2022,shuffle=True)
    i=0
    for train_index, test_index in kf.split(cells):
            i=i+1
            result_folder = args.result_folder + '/CV10_cold_cell/fold'+str(i)                
            train_val, test_set = cells.iloc[train_index], cells.iloc[test_index]
            train_set, val_set = train_test_split(train_val, test_size = 1/9, random_state = 0)
            train_set = pd.merge(train_set,CDR,on='COSMIC_ID')
            val_set = pd.merge(val_set,CDR,on='COSMIC_ID')
            test_set = pd.merge(test_set,CDR,on='COSMIC_ID')
            if os.path.exists(result_folder): 
                train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
                test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
                val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)
            else:
                os.makedirs(result_folder)
                train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
                test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
                val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)

if args.scenarios == 'cold scaffold':
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    index = CDR.index.values
    smiles_list = CDR['smiles']
    sort=False
    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)
    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    all_scaffolds_list = list(all_scaffolds.keys())
    # CV10 split
    kf = KFold(n_splits=10,random_state=2022,shuffle=True)
    i=0
    for train_index, test_index in kf.split(all_scaffolds_list):
        i=i+1
        result_folder = args.result_folder + '/CV10_cold_scaffold/fold'+str(i)
        train_index, val_index = train_test_split(train_index, test_size = 1/9, random_state = 0)
        train_set = []
        test_set = []
        val_set = []
        for j in train_index:
            train_set.append(CDR.iloc[all_scaffolds[all_scaffolds_list[j]]])
        for j in test_index:
            test_set.append(CDR.iloc[all_scaffolds[all_scaffolds_list[j]]])
        for j in val_index:
            val_set.append(CDR.iloc[all_scaffolds[all_scaffolds_list[j]]])
        train_set = pd.concat(train_set,axis=0)
        test_set = pd.concat(test_set,axis=0)
        val_set = pd.concat(val_set,axis=0)
        if os.path.exists(result_folder): 
            train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
            test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
            val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)
        else:
            os.makedirs(result_folder)
            train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
            test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
            val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)

if args.scenarios == 'cold cold cluster':
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    # get cell info
    CDR = pd.read_csv('./data/GDSC/data_processed/CDR_n156813.txt',sep='\t',index_col=0)
    cell_info = CDR[['COSMIC_ID','cell_type','assay_name']].drop_duplicates(subset=['cell_type'])
    cell_info['COSMIC_ID'] = [int(item) for item in cell_info['COSMIC_ID']]
    # get omics data
    rna_data = pd.read_csv('./data/GDSC/data_processed/RNA_n18451_1018_zscore.csv',index_col=0)
    rna_data.columns = [name.split('.')[0][1:] for name in rna_data.columns.values]
    rna_data = rna_data.T
    genetic = pd.read_csv('./data/GDSC/data_processed/Genetic_features_n969_735.txt',sep='\t',index_col=0)
    mrna = pd.read_csv('./data/GDSC/data_processed/mrna_n20617_1028_zscore.csv',index_col=0)
    mrna = mrna.T

    rna_data2 = rna_data.loc[cell_info['assay_name']]
    genetic2 = genetic.loc[cell_info['COSMIC_ID']]
    mrna2 = mrna.loc[cell_info['cell_type']]

    rna_data2 = rna_data2.reset_index(drop=True)
    genetic2 = genetic2.reset_index(drop=True)
    mrna2 = mrna2.reset_index(drop=True)

    data = pd.concat([rna_data2,genetic2,mrna2],axis=1)
    data.index = cell_info['cell_type']
    # data scale
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # kmeans
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
    kmeans.fit(data_scaled)

    # get cluster labels
    labels = kmeans.labels_
    # print(labels)
    labels = pd.DataFrame(labels)
    labels.columns = ['cell_label']
    labels['cell_type'] = cell_info['cell_type'].values
    # labels['label'].value_counts().to_csv('./data/GDSC/data_processed/cell_k-means.csv')

    # data split based on cell cluster
    CDR = pd.merge(CDR,labels,on='cell_type')
    kf = KFold(n_splits=10,random_state=2022,shuffle=True)
    i=0
    label = labels.drop_duplicates(subset=['cell_label'])
    for train_index, test_index in kf.split(label):
        i=i+1
        result_folder = args.result_folder + '/CV10_cold_cell_cluster'+ args.n_clusters+ '/fold'+str(i)                
        train_val, test_set = label.iloc[train_index], label.iloc[test_index]
        train_set, val_set = train_test_split(train_val, test_size = 1/9, random_state = 0)
        train_cell_label = pd.merge(labels['cell_label'],train_set,on='cell_label')
        test_cell_label = pd.merge(labels['cell_label'],test_set,on='cell_label')
        val_cell_label = pd.merge(labels['cell_label'],val_set,on='cell_label')
        train_set = pd.merge(train_set['cell_label'],CDR,on='cell_label')
        val_set = pd.merge(val_set['cell_label'],CDR,on='cell_label')
        test_set = pd.merge(test_set['cell_label'],CDR,on='cell_label')
        if os.path.exists(result_folder): 
            train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
            test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
            val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)
        else:
            os.makedirs(result_folder)
            train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
            test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
            val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)

if args.scenarios == 'cold cell & drug':
    drugs = pd.DataFrame({'drug_id':CDR['drug_id'].unique()}) 
    cells = pd.DataFrame({'cell_type':CDR['cell_type'].unique()}) 
    kf = KFold(n_splits=10,random_state=2022,shuffle=True)
    i=0
    for d_train_index, d_test_index in kf.split(drugs):
        i=i+1
        print(i)
        result_folder = args.result_folder + '/CV10_cold_cell_&_drug/fold'+str(i)
        d_train_val, d_test_set = drugs.iloc[d_train_index], drugs.iloc[d_test_index]
        d_train_set, d_val_set = train_test_split(d_train_val, test_size = 1/9, random_state = 0)
        if os.path.exists(result_folder): 
            d_train_set.to_csv(result_folder+'/d_train.txt',sep='\t',index=False)
            d_test_set.to_csv(result_folder+'/d_test.txt',sep='\t',index=False)
            d_val_set.to_csv(result_folder+'/d_val.txt',sep='\t',index=False)
        else:
            os.makedirs(result_folder)
            d_train_set.to_csv(result_folder+'/d_train.txt',sep='\t',index=False)
            d_test_set.to_csv(result_folder+'/d_test.txt',sep='\t',index=False)
            d_val_set.to_csv(result_folder+'/d_val.txt',sep='\t',index=False)       
    i=0
    for c_train_index, c_test_index in kf.split(cells):
        i=i+1
        print(i)
        result_folder = args.result_folder + '/CV10_cold_cell_&_drug/fold'+str(i)
        c_train_val, c_test_set = cells.iloc[c_train_index], cells.iloc[c_test_index]
        c_train_set, c_val_set = train_test_split(c_train_val, test_size = 1/9, random_state = 0)
        if os.path.exists(result_folder): 
            c_train_set.to_csv(result_folder+'/c_train.txt',sep='\t',index=False)
            c_test_set.to_csv(result_folder+'/c_test.txt',sep='\t',index=False)
            c_val_set.to_csv(result_folder+'/c_val.txt',sep='\t',index=False)
        else:
            os.makedirs(result_folder)
            c_train_set.to_csv(result_folder+'/c_train.txt',sep='\t',index=False)
            c_test_set.to_csv(result_folder+'/c_test.txt',sep='\t',index=False)
            c_val_set.to_csv(result_folder+'/c_val.txt',sep='\t',index=False)
    for i in range(1,11):
        result_folder = args.result_folder + '/CV10_cold_cell_&_drug/fold'+str(i)
        d_train_set = pd.read_csv(result_folder+'/d_train.txt',sep='\t')
        d_test_set = pd.read_csv(result_folder+'/d_test.txt',sep='\t')
        d_val_set = pd.read_csv(result_folder+'/d_val.txt',sep='\t')
        c_train_set = pd.read_csv(result_folder+'/c_train.txt',sep='\t')
        c_test_set = pd.read_csv(result_folder+'/c_test.txt',sep='\t')
        c_val_set = pd.read_csv(result_folder+'/c_val.txt',sep='\t')
        train_set = pd.merge(CDR,d_train_set,on='drug_id')
        train_set = pd.merge(train_set,c_train_set,on='cell_type')
        test_set = pd.merge(CDR,d_test_set,on='drug_id')
        test_set = pd.merge(test_set,c_test_set,on='cell_type')
        val_set = pd.merge(CDR,d_val_set,on='drug_id')
        val_set = pd.merge(val_set,c_val_set,on='cell_type')
        if os.path.exists(result_folder): 
            train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
            test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
            val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)
        else:
            os.makedirs(result_folder)
            train_set.to_csv(result_folder+'/train.txt',sep='\t',index=False)
            test_set.to_csv(result_folder+'/test.txt',sep='\t',index=False)
            val_set.to_csv(result_folder+'/val.txt',sep='\t',index=False)

         
if args.scenarios == 'cold cell & scaffold':
    for i in range(1,11):
        _foresultlder = args.result_folder + '/cold_cell_cluster10_&_scaffold/fold'+str(i)
        train = pd.read_csv('./data/GDSC/CV10_cold_scaffold/fold'+str(i)+'/train.txt',sep='\t',index_col=False)
        test = pd.read_csv('./data/GDSC/CV10_cold_scaffold/fold'+str(i)+'/test.txt',sep='\t',index_col=False)
        val = pd.read_csv('./data/GDSC/CV10_cold_scaffold/fold'+str(i)+'/val.txt',sep='\t',index_col=False)
        cell_train = pd.read_csv('./data/GDSC/CV10_cold_cell_cluster10/fold'+str(i)+'/train.txt',sep='\t',index_col=False)
        cell_test = pd.read_csv('./data/GDSC/CV10_cold_cell_cluster10/fold'+str(i)+'/test.txt',sep='\t',index_col=False)
        cell_val = pd.read_csv('./data/GDSC/CV10_cold_cell_cluster10/fold'+str(i)+'/val.txt',sep='\t',index_col=False)
        cell_train = cell_train.drop_duplicates(subset=['cell_type'])
        cell_test = cell_test.drop_duplicates(subset=['cell_type'])
        cell_val = cell_val.drop_duplicates(subset=['cell_type'])
        train = pd.merge(train,cell_train['cell_type'],on='cell_type')
        test = pd.merge(test,cell_test['cell_type'],on='cell_type')
        val = pd.merge(val,cell_val['cell_type'],on='cell_type')
        if os.path.exists(result_folder): 
            train.to_csv(result_folder+'/train.txt',sep='\t',index=False)
            test.to_csv(result_folder+'/test.txt',sep='\t',index=False)
            val.to_csv(result_folder+'/val.txt',sep='\t',index=False)
        else:    
            os.makedirs(result_folder)
            train.to_csv(result_folder+'/train.txt',sep='\t',index=False)
            test.to_csv(result_folder+'/test.txt',sep='\t',index=False)
            val.to_csv(result_folder+'/val.txt',sep='\t',index=False)


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain assert from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold













