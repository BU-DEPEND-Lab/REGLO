python train.py --gpu_id 2 --dataset twitter_spam_account --training_mode clean --epoch 1000 --batch_size 1024 &&
python train.py --gpu_id 2 --dataset twitter_spam_account --training_mode local_adv --eps_train 0.2 --eps_test 0.1 --epoch 1000 --batch_size 1024 &&
python train.py --gpu_id 2 --dataset twitter_spam_account --training_mode global_adv --eps_train 0.2 --eps_test 0.1 --global_robustness_output_bound 50 --epoch 1000 --batch_size 1024
