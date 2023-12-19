#!/bin/bash
python reglo_german.py --mode 'clean' --model_name '/20220813-200209/model_german_credit_epoch_200_batch_64_lr_0.0001' \
						--age_range_lb 65 --age_range_ub 75  &&  ##clean
python reglo_german.py --mode 'local_adv' --model_name '/20220813-200238/model_german_credit_pgd_eps_0.3_0.2_steps_10_200_epoch_200_batch_1024_lr_0.0001' \
						--age_range_lb 65 --age_range_ub 75  &&  ##local_adv
python reglo_german.py --mode 'global_adv' --model_name '/20220815-65-75/model_german_credit_pgd_eps_0.3_0.2_steps_10_200_output_bound_0.5_loss_coeff_0.1_points_100_1000_epoch_200_batch_1024_lr_0.0003' \
						--age_range_lb 65 --age_range_ub 75  && ##global_adv
python reglo_german.py --mode 'ST-AT-G_fine_tuning' --model_name '/20220815-65-75/model_german_credit_fine_tuned_pgd_eps_0.3_0.2_steps_10_200_output_bound_60.0_loss_coeff_0.1_points_100_1000_epoch_100_lr_0.0001' \
						--age_range_lb 65 --age_range_ub 75  && ##ST + ATG
python reglo_german.py --mode 'AT-AT-G_fine_tuning' --model_name '/20220815-65-75/model_german_credit_fine_tuned_pgd_eps_0.3_0.2_steps_10_200_output_bound_60.0_loss_coeff_0.1_points_100_1000_epoch_100_lr_0.0001' \
						--age_range_lb 65 --age_range_ub 75     ##AT + ATG