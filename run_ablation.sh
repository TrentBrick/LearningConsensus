python3 run.py --exp_name ablation_analy \
--random_seed 27 13 11 --epochs 700 --iters_per_epoch 20 100 250 \
--max_round_len 25 --print_every 500 --temp_anneal 0.985 \
--send_all_first_round_reward 0.0 0.3 --consistency_violation -1.0 \
--validity_violation -2.0 -1.0 --majority_violation -1.0 0.0