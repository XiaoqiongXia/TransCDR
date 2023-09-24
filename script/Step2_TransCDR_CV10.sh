cd ./TransCDR
python Step2_train_model.py \
--model_type 'regression' \
--data_path './data/GDSC/CV10' \
--omics 'expr + mutation + methylation' \
--input_dim_drug 2092 \
--lr 1e-5 \
--BATCH_SIZE 64 \
--train_epoch 100 \
--pre_train 'True' \
--screening 'None' \
--fusion_type 'encoder' \
--drug_encoder 'None' \
--drug_model 'sequence + graph + FP' \
--modeldir './result/CV10' \
--seq_model 'seyonec/PubChem10M_SMILES_BPE_450k' \
--graph_model 'gin_supervised_masking' \
--external_dataset 'None'
