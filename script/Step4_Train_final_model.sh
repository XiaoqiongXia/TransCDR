cd ./TransCDR
python Step4_Train_final_model.py \
--CV10_result_path '/home/u19111510052/research/Drug_Cancer/Drug_Cancer/result/cross_att_clas_regr/CV10_cold_cell_cold_drug' \
--data_path './data/GDSC/CV10' \
--model_type 'regression' \
--modeldir './result/Final_model/regression'
