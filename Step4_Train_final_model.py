import pandas as pd
from DataEncoding import DataEncoding
from model import TransCDR
import numpy as np
import math
import argparse
parser = argparse.ArgumentParser(description='training the final model')
parser.add_argument('--CV10_result_path', type=str, required=True, help='CV10_result_path')
parser.add_argument('--data_path', type=str, required=True, help='data used for training TransCDR')
parser.add_argument('--model_type', type=str, required=True, help='classification or regression')
parser.add_argument('--modeldir', type=str, required=True, help='the result path of final model')
args = parser.parse_args()

# get the best epoch from CV10 results
def get_best_epoch(path,model_type):
    best_epoch = []
    if model_type == 'regression':
        for i in range(1,11):
            result_folder =path+'/fold'+str(i)+'/'+'/valid_markdowntable.txt'
            res = pd.read_csv(result_folder)
            RMSE = []
            for j in range(2,len(res)-1):
                RMSE.append(float(res.iloc[j,0].split('|')[2]))
            best_epoch.append(pd.DataFrame(RMSE).rank(ascending=False).iloc[-1].values[0])
    if model_type == 'classification':
        for i in range(1,11):
                result_folder =path+'/fold'+str(i)+'/valid_markdowntable.txt'
                res = pd.read_csv(result_folder)
                roc = []
                for j in range(2,len(res)-1):
                    roc.append(float(res.iloc[j,0].split('|')[2]))
                best_epoch.append(pd.DataFrame(roc).rank().iloc[-1].values[0])
    best_epoch = math.ceil(np.mean(best_epoch))
    return best_epoch

# train final models
best_epoch = get_best_epoch(args.CV10_result_path,args.model_type)
i=1
train = pd.read_csv(args.data_path+'/fold'+str(i)+'/train.txt',sep='\t')
test = pd.read_csv(args.data_path+'/fold'+str(i)+'/test.txt',sep='\t')
val = pd.read_csv(args.data_path+'/fold'+str(i)+'/val.txt',sep='\t')

config = {
        'model_type':args.model_type, # classification, regression
        'omics':'expr + mutation + methylation', # expr + mutation + methylation
        'input_dim_drug':300+768+1024,  # graph:300, seq:768, sequence + graph:1068
        'input_dim_rna':18451,
        'input_dim_genetic':735,
        'input_dim_mrna':20617,
        'KG':'',
        'lr':1e-5,
        'decay':0,
        'BATCH_SIZE':64,
        'train_epoch':best_epoch,
        'pre_train':True,
        'screening':'None',
        'fusion_type':'encoder', # concat, decoder,encoder
        'drug_encoder':'', # CNN, RNN, Transformer, GCN, NeuralFP,AttentiveFP
        'drug_model': 'sequence + graph + FP', # graph, sequence, sequence + graph FP
        'modeldir':args.modeldir,
        'seq_model':'seyonec/ChemBERTa-zinc-base-v1',
        'graph_model':'gin_supervised_masking',
        'external_dataset':'None'
    }
data_encoding = DataEncoding(**config)
train_set, test_set, val_set = data_encoding.encode(train, test, val)
df = pd.concat([train_set,test_set,val_set])
df.reset_index(inplace = True,drop=True)
net = TransCDR(**config)
net.train(train_drug=df,
        test_drug=None,
        val_drug=None)
net.save_model()
if args.model_type == 'regression':
    y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = net.predict(df)
if args.model_type == 'classification':
    auc, pr,f1,loss,y_pred,y_label = net.predict(df)

pd.DataFrame(y_pred).to_csv(args.modeldir+'/y_pred.csv')
pd.DataFrame(y_label).to_csv(args.modeldir+'/y_label.csv')

