cd ./TransCDR
python Step1_Data_split.py \
--model_type "regression" \
--scenarios "warm start" \
--n_clusters 0 \
--n_sampling 0 \
--result_folder "./data/GDSC"
