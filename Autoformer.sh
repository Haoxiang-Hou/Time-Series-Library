# export CUDA_VISIBLE_DEVICES=7

model_name=Autoformer

python -u run.py \
  --train_epochs 2 \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id CMEES_96_96 \
  --batch_size 100 \
  --learning_rate 0.0001 \
  --model $model_name \
  --data CMEES \
  --features M \
  --freq_per_second 1000 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 15 \
  --dec_in 15 \
  --c_out 15 \
  --target_size 8 \
  --des 'Exp' \
  --itr 1 \
  --setting test3 \
  --rolling_window_stride 100000 \
  --d_model 64 \
  --d_ff 256 \
  --n_heads 4 \
  --moving_avg 25 \
  --dataset_class 'FeatureDataset'


  

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --itr 1