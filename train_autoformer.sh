#!/bin/bash

set -e
set -u
set -o pipefail
# set -x


task_name=long_term_forecast
is_training=1
model_name=Autoformer
data=CMEES
features=M
enc_in=1
dec_in=1
c_out=1
des='Exp'
itr=1

freq_per_second=100

seq_len=96
label_len=48
pred_len=96

d_model=64
d_ff=256
n_heads=4
e_layers=2
d_layers=1
factor=3
moving_avg=25

model_id=CMEES_96_96
exp_name=test

train_epochs=10
rolling_window_stride=100
learning_rate=0.0001
batch_size=5000

for freq_per_second in 1000 100 10; do
    for moving_avg in 5 25 ; do
        for seq_len in 64 128; do
            label_len=$((seq_len / 2))
            for pred_len in 64 128 ; do
                for learning_rate in 5e-3 1e-3 5e-4 1e-4 5e-5; do

                    exp_name=${data}_${model_name}_freq${freq_per_second}_moving_avg${moving_avg}_seq${seq_len}_label${label_len}_pred${pred_len}_lr${learning_rate}
                    model_id=${exp_name}
                    # exp_name=test

                    echo python -u run.py --task_name $task_name --is_training $is_training --model $model_name --data $data --features $features --enc_in $enc_in --dec_in $dec_in --c_out $c_out --des "$des" --itr $itr --freq_per_second $freq_per_second --seq_len $seq_len --label_len $label_len --pred_len $pred_len --d_model $d_model --d_ff $d_ff --n_heads $n_heads --e_layers $e_layers --d_layers $d_layers --factor $factor --moving_avg $moving_avg --model_id $model_id --setting $exp_name --train_epochs $train_epochs --rolling_window_stride $rolling_window_stride --learning_rate $learning_rate --batch_size $batch_size
                    python -u run.py --task_name $task_name --is_training $is_training --model $model_name --data $data --features $features --enc_in $enc_in --dec_in $dec_in --c_out $c_out --des "$des" --itr $itr --freq_per_second $freq_per_second --seq_len $seq_len --label_len $label_len --pred_len $pred_len --d_model $d_model --d_ff $d_ff --n_heads $n_heads --e_layers $e_layers --d_layers $d_layers --factor $factor --moving_avg $moving_avg --model_id $model_id --setting $exp_name --train_epochs $train_epochs --rolling_window_stride $rolling_window_stride --learning_rate $learning_rate --batch_size $batch_size
                done
            done
        done
    done
done

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --model_id CMEES_96_96 \
#   --batch_size 100 \
#   --learning_rate 0.0001 \
#   --model Autoformer \
#   --data CMEES \
#   --features M \
#   --freq_per_second 100 \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --des 'Exp' \
#   --itr 1 \
#   --setting test

# or model_name in MultiLayerSelfAttentionRespectiveResidualRoPEPosition; do
#     for history_seq_len in 100; do
#         batch_rolling_window_stride=100
#         for hidden_dim in 64 128; do
#             for num_layers in 2 4 8 16 32; do
#                 forward_dim=$((hidden_dim * 4))
#                 for learning_rate in 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4; do

#                     exp_name=${dataset}_history-seq${history_seq_len}_${model_name}_hidden-dim${hidden_dim}_forward-dim${forward_dim}_num-layers${num_layers}_lr${learning_rate}