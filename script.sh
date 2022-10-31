# python train.py --gpus 6 7 --mode train --config_path config_76.yaml --name 76_mean_std > 76_mean_std.txt
# python train.py --gpus 6 7 --mode train --config_path config_15.yaml --name 15_mean_std > 15_mean_std.txt
# python train.py --gpus 6 7 --mode train --config_path config_34.yaml --name 34_mean_std > 34_mean_std.txt
# python train.py --gpus 0 1 6 7 --mode train  --config_path config_76.yaml --checkpoint initial.ckpt  --name=for_TF_76_UC --freeze=upconv
# python train.py --gpus 0 1 6 7 --mode train  --config_path config_15.yaml --checkpoint initial.ckpt  --name=for_TF_15_UC --freeze=upconv
# python train.py --gpus 0 1 6 7 --mode train  --config_path config_34.yaml --checkpoint initial.ckpt  --name=for_TF_34_UC --freeze=upconv

# python train.py --gpus 6 7 --mode train  --config_path config_34_.yaml --checkpoint initial.ckpt  --name=for_TF_34_FF --freeze=~film_final

python train.py --gpus 4 5 6 7 --mode train  --config_path config_15_.yaml --checkpoint initial.ckpt  --name=upconv_15_1e-5 --freeze=upconv
python train.py --gpus 4 5 6 7 --mode train  --config_path config_34_.yaml --checkpoint initial.ckpt  --name=upconv_34_1e-5 --freeze=upconv
python train.py --gpus 4 5 6 7 --mode train  --config_path config_76_.yaml --checkpoint initial.ckpt  --name=upconv_76_1e-5 --freeze=upconv