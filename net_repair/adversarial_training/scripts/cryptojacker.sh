python train.py --gpu_id 1 --dataset cryptojacker --training_mode clean --epoch 1000 --batch_size 1024 &&
python train.py --gpu_id 1 --dataset cryptojacker --training_mode local_adv --eps_train 0.3 --eps_test 0.2 --epoch 1000 --batch_size 1024 &&
python train.py --gpu_id 1 --dataset cryptojacker --training_mode global_adv --eps_train 0.3 --eps_test 0.2 --global_robustness_output_bound 0.5 --epoch 1000 --batch_size 1024
