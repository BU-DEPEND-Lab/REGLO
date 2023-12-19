python train.py --gpu_id 3 --dataset auto_mpg --training_mode clean --epoch 100 --batch_size 32 &&
python train.py --gpu_id 3 --dataset auto_mpg --training_mode local_adv --eps_train 0.05 --eps_test 0.05 --epoch 100 --batch_size 32 &&
python train.py --gpu_id 3 --dataset auto_mpg --training_mode global_adv --eps_train 0.05 --eps_test 0.05 --global_robustness_output_bound 1.5 --epoch 100 --batch_size 32
