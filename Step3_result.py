import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='CV10 results')
parser.add_argument('--CV10_result_path', type=str, required=True, help='the CV10 result path')
args = parser.parse_args()
# CV10 results
MSE = []
RMSE = []
Pearson = []
Pearson_pval = []
Spearman = []
Spearman_pval = []
Concordance_Index = []
for i in range(1,11):
    res = pd.read_csv(args.CV10_result_path+'/fold' + str(i) + '/test_markdowntable.txt')
    MSE.append(float(res.iloc[2,0].split('|')[1]))
    RMSE.append(float(res.iloc[2,0].split('|')[2]))
    Pearson.append(float(res.iloc[2,0].split('|')[3]))
    Pearson_pval.append(float(res.iloc[2,0].split('|')[4]))
    Spearman.append(float(res.iloc[2,0].split('|')[5]))
    Spearman_pval.append(float(res.iloc[2,0].split('|')[6]))
    Concordance_Index.append(float(res.iloc[2,0].split('|')[7]))

result = pd.DataFrame({'MSE':MSE,
                        'RMSE':RMSE,
                        'Pearson':Pearson,
                        'Pearson_pval':Pearson_pval,
                        'Spearman':Spearman,
                        'Spearman_pval':Spearman_pval,
                        'Concordance_Index':Concordance_Index})
result = result.sort_values(by='RMSE')
res = result.describe()
res.to_csv(args.CV10_result_path+'/res_mean.csv')
result.to_csv(args.CV10_result_path+ '/res.csv')


