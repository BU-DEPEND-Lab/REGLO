python train.py --gpu_id 0 --dataset twitter_spam_url --training_mode clean --epoch 1000 --batch_size 1024 &&
python train.py --gpu_id 0 --dataset twitter_spam_url --training_mode local_adv --eps_train 1.6 --eps_test 1.5 --epoch 1000 --batch_size 1024 &&
python train.py --gpu_id 0 --dataset twitter_spam_url --training_mode global_adv --eps_train 1.6 --eps_test 1.5 --global_robustness_output_bound 10 --epoch 1000 --batch_size 1024
