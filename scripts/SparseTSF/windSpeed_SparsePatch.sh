if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SparseTSFPatch

root_path_name=./dataset/wind
data_path_name=Turb1.csv
model_id_name=windSpeed
data_name=custom

seq_len=432
for pred_len in 36 72 144
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features MS \
    --target Wspd \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 18 \
    --d_model 90 \
    --dropout 0.2 \
    --enc_in 10 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 64 --learning_rate 0.02
done
