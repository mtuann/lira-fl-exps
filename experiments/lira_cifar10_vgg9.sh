python tuan_simulated_averaging.py \
--batch-size 64 \
--test-batch-size 2560 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 1500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg9 \
--fl_mode fixed-freq \
--attacker_pool_size 0 \
--defense_method no-defense \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 1 \
--stddev 0.025 \ 
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--baseline False \
--instance baseline_K_10_N_200_Cifar10 \
--norm_bound 2 \
--atk_model_train_epoch 10 \
--num_dps_attacker 1000 \
--attack_alpha 0.5 \
--atk_eps 0.3 \
--attack_portion 1.0 \
--scale 1.0 \
--target_label 1 \
--atk_lr 0.001 \
--device=cuda