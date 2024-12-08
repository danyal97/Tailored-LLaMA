prune_ckpt_path='taylor/param_first/group_reduction/sum/0.9'
tune_ckpt_path='taylor/param_first/group_reduction/sum/0.9'

echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=3 python hf_prune.py --base_model baffo32/decapoda-research-llama-7B-hf --pruning_ratio 0.95 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_before_train --test_after_train --taylor param_first --save_model --grouping_strategy sum
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=3 python post_training.py --base_model baffo32/decapoda-research-llama-7B-hf --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project taylor_param_first_group_reduction_sum --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"



