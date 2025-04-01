if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate paper8

model_name=TMN

python -u run_exp.py \
  --root_path ./datasets/tablet \
  --data tablet \
  --model $model_name \
  --itr 5 --loss mse \
  --patience 100 \
  --lr_sch_gamma 0.99 \
  --batch_size 32 --learning_rate 0.005 --weight_decay 0.05 \
  --aci_token 12 --aci_dp 0.1 --aci_nhead 5 --aci_ff 512 --label_norm True --norm_lr 0.01 \
  --aci_ref_n 36 --aci_fc1 256 --aci_fc2 64 \
  --train_d 1 --target_d 1 \
  --individual >logs/$model_name'_I_'tablet.log

python -u run_exp.py \
  --root_path ./datasets/tablet \
  --data tablet \
  --model $model_name \
  --itr 5 --loss mse \
  --patience 100 \
  --lr_sch_gamma 0.99 \
  --batch_size 32 --learning_rate 0.005 --weight_decay 0.05 \
  --aci_token 12 --aci_dp 0.1 --aci_nhead 5 --aci_ff 512 --label_norm True --norm_lr 0.01 \
  --aci_ref_n 36 --aci_fc1 256 --aci_fc2 64 \
  --train_d 1 --target_d 2 \
  --individual >logs/$model_name'_I_'tablet.log

python -u run_exp.py \
  --root_path ./datasets/tablet \
  --data tablet \
  --model $model_name \
  --itr 5 --loss mse \
  --patience 100 \
  --lr_sch_gamma 0.99 \
  --batch_size 32 --learning_rate 0.005 --weight_decay 0.05 \
  --aci_token 12 --aci_dp 0.1 --aci_nhead 5 --aci_ff 512 --label_norm True --norm_lr 0.01 \
  --aci_ref_n 36 --aci_fc1 256 --aci_fc2 64 \
  --train_d 2 --target_d 2 \
  --individual >logs/$model_name'_I_'tablet.log

python -u run_exp.py \
  --root_path ./datasets/tablet \
  --data tablet \
  --model $model_name \
  --itr 5 --loss mse \
  --patience 100 \
  --lr_sch_gamma 0.99 \
  --batch_size 32 --learning_rate 0.005 --weight_decay 0.05 \
  --aci_token 12 --aci_dp 0.1 --aci_nhead 5 --aci_ff 512 --label_norm True --norm_lr 0.01 \
  --aci_ref_n 36 --aci_fc1 256 --aci_fc2 64 \
  --train_d 2 --target_d 1 \
  --individual >logs/$model_name'_I_'tablet.log

python -u run_exp.py \
  --root_path ./datasets/mangoDMC \
  --data mango_dmc \
  --model $model_name \
  --itr 5 --loss mse \
  --patience 100 \
  --lr_sch_gamma 0.99 \
  --batch_size 32 --learning_rate 0.001 --weight_decay 0.01 \
  --aci_token 12 --aci_dp 0.1 --aci_nhead 5 --aci_ff 512 --label_norm True --norm_lr 0.01 \
  --aci_ref_n 36 --aci_fc1 256 --aci_fc2 64 \
  --individual >logs/$model_name'_I_'mango_dmc.log

python -u run_exp.py \
  --root_path ./datasets/strawberryPuree \
  --data strawberry_puree \
  --model $model_name \
  --itr 5 --loss mse \
  --patience 100 \
  --lr_sch_gamma 0.99 \
  --batch_size 32 --learning_rate 0.001 --weight_decay 0.01 \
  --aci_token 12 --aci_dp 0.1 --aci_nhead 5 --aci_ff 512 --label_norm True --norm_lr 0.01 \
  --aci_ref_n 36 --aci_fc1 256 --aci_fc2 64 \
  --individual >logs/$model_name'_I_'tablet.log

python -u run_exp.py \
  --root_path ./datasets/Melamine \
  --data melamine \
  --model $model_name \
  --itr 5 --loss mse \
  --patience 100 \
  --lr_sch_gamma 0.99 \
  --batch_size 32 --learning_rate 0.002 --weight_decay 0 \
  --aci_token 2 --aci_dp 0 --aci_nhead 1 --aci_ff 128 --label_norm True --norm_lr 0.001 \
  --aci_ref_n 36 --aci_fc1 64 --aci_fc2 16 \
  --train_d R562 --target_d R568 \
  --individual >logs/$model_name'_I_'tablet.log

python -u run_exp.py \
  --root_path ./datasets/ \
  --data apple_leaf \
  --model $model_name \
  --itr 5 --loss mse \
  --patience 100 --train_epochs 300 \
  --lr_sch_gamma 0.99 \
  --batch_size 32 --learning_rate 0.001 --weight_decay 0.001 \
  --aci_token 22 --aci_dp 0 --aci_nhead 1 --aci_ff 256 --label_norm True --norm_lr 0.001 \
  --aci_ref_n 36 --aci_fc1 128 --aci_fc2 32 --aci_stride 5 \
  --individual >logs/$model_name'_I_'tablet.log

python -u run_exp.py \
  --root_path ./datasets/ \
  --data rruff \
  --model $model_name \
  --itr 5 --loss mse \
  --patience 100 --train_epochs 300 \
  --lr_sch_gamma 0.99 \
  --batch_size 32 --learning_rate 0.001 --weight_decay 0 \
  --aci_token 12 --aci_dp 0 --aci_nhead 1 --aci_ff 256 --label_norm True --norm_lr 0.001 \
  --aci_ref_n 36 --aci_fc1 256 --aci_fc2 64 --aci_stride 1 \
  --individual >logs/$model_name'_I_'tablet.log
