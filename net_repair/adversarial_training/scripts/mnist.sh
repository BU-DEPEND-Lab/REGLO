python train.py --gpu_id 3 --dataset mnist --training_mode clean --epoch 100 --batch_size 256 &&
python train.py --gpu_id 3 --dataset mnist --training_mode local_adv --eps_train 0.3 --eps_test 0.2 --epoch 100 --batch_size 256 &&
python train.py --gpu_id 3 --dataset mnist --training_mode global_adv --eps_train 0.3 --eps_test 0.2 --global_robustness_output_bound 60 --epoch 100 --batch_size 256
