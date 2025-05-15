#!/bin/bash

set -e
set -u
set -o pipefail
# set -x

export CUDA_VISIBLE_DEVICES=8

task_name=long_term_forecast
is_training=1
model_name=Autoformer
data=CMEES
dataset_class=LastHistoryMidPriceSeriesDataset
features=M
des='Exp'
itr=1

enc_in=1
dec_in=1
c_out=1
target_size=1
freq_per_second=100
loss_function='MaskedMSELoss'

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
train_rolling_window_stride=100
dev_rolling_window_stride=100
learning_rate=0.0001
batch_size=3500

for freq_per_second in 100 10; do
    for moving_avg in 5 25 ; do
        for seq_len in 96; do
            label_len=$((seq_len / 2))
            for pred_len in 96 ; do
                for learning_rate in 5e-3 1e-3 5e-4 1e-4; do

                    exp_name=${data}_${dataset_class}_${model_name}_freq${freq_per_second}_moving_avg${moving_avg}_seq${seq_len}_label${label_len}_pred${pred_len}_${loss_function}_lr${learning_rate}
                    model_id=${exp_name}
                    # exp_name=test

                    echo python -u run.py --task_name $task_name --is_training $is_training --model $model_name --data $data --dataset_class $dataset_class --features $features --enc_in $enc_in --dec_in $dec_in --c_out $c_out --target_size $target_size --loss_function $loss_function --des "$des" --itr $itr --freq_per_second $freq_per_second --seq_len $seq_len --label_len $label_len --pred_len $pred_len --d_model $d_model --d_ff $d_ff --n_heads $n_heads --e_layers $e_layers --d_layers $d_layers --factor $factor --moving_avg $moving_avg --model_id $model_id --setting $exp_name --train_epochs $train_epochs --train_rolling_window_stride $train_rolling_window_stride --dev_rolling_window_stride $dev_rolling_window_stride --learning_rate $learning_rate --batch_size $batch_size
                    # srun -p a10,4090 --gres=gpu:1 -c 8 --mem-per-cpu=10G --qos qnormal 
                    python -u run.py --task_name $task_name --is_training $is_training --model $model_name --data $data --dataset_class $dataset_class --features $features --enc_in $enc_in --dec_in $dec_in --c_out $c_out --target_size $target_size --loss_function $loss_function --des "$des" --itr $itr --freq_per_second $freq_per_second --seq_len $seq_len --label_len $label_len --pred_len $pred_len --d_model $d_model --d_ff $d_ff --n_heads $n_heads --e_layers $e_layers --d_layers $d_layers --factor $factor --moving_avg $moving_avg --model_id $model_id --setting $exp_name --train_epochs $train_epochs --train_rolling_window_stride $train_rolling_window_stride --dev_rolling_window_stride $dev_rolling_window_stride --learning_rate $learning_rate --batch_size $batch_size 
                done
            done
        done
    done
done