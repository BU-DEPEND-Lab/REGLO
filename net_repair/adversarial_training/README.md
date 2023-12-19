# Normal and Local Adversarial Training

## Running scripts

```
./scripts/mnist.sh  # MNIST training

./scripts/cryptojacker.sh  # Cryptokacker training

./scripts/twitter_spam_account.sh  # Twitter Spam Account Training

./scripts/twitter_spam_ulr.sh  # Twitter Spam URL Training

./scripts/german_credits.sh  # German Credit Training
```

## MNIST

### Normal Training
```
python train.py --gpu_id 0 --dataset mnist --training_mode clean --epoch 100 --batch_size 256
```

### Local Adversarial Training
```
python train.py --gpu_id 0 --dataset mnist --training_mode local_adv --eps_train 0.3 --eps_test 0.2 --epoch 100 --batch_size 256
```

## Cryptojacker

### Normal Training
```
python train.py --gpu_id 0 --dataset cryptojacker --training_mode clean --epoch 1000 --batch_size 1024
```

### Local Adversarial Training
```
python train.py --gpu_id 0 --dataset cryptojacker --training_mode local_adv --eps_train 0.3 --eps_test 0.2 --epoch 1000 --batch_size 1024
```

## Twitter Spam Account

### Normal Training
```
python train.py --gpu_id 0 --dataset twitter_spam_account --training_mode clean --epoch 1000 --batch_size 1024
```

### Local Adversarial Training
```
python train.py --gpu_id 0 --dataset twitter_spam_account --training_mode local_adv --eps_train 0.2 --eps_test 0.1 --epoch 1000 --batch_size 1024
```

## Twitter Spam URL

### Normal Training
```
python train.py --gpu_id 0 --dataset twitter_spam_url --training_mode clean --epoch 1000 --batch_size 1024
```

### Local Adversarial Training
```
python train.py --gpu_id 0 --dataset twitter_spam_url --training_mode local_adv --eps_train 1.6 --eps_test 1.5 --epoch 1000 --batch_size 1024
```

## Result

The trained model and traning log are stored in `model` folder.

For example, to view the log of MNIST local adversarial training, run:
```
tensorboard --logdir ./model/mnist_local_adv
```

## PGD evaluation of Global Robustness

```
python eval.py --dataset mnist --input_bound 0.3 --output_bound 10 --number_of_points 100
```
