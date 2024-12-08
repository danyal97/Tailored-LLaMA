prune_ckpt_path='taylor/param_first/group_reduction/sum'
tune_ckpt_path='taylor/param_first/group_reduction/sum/piqa'

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=3 python post_training.py --base_model baffo32/decapoda-research-llama-7B-hf --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path piqa --output_dir tune_log/$tune_ckpt_path --wandb_project piqa_taylor_param_first_group_reduction_sum --lora_r 8 --num_epochs 8 --learning_rate 1e-4 --batch_size 16
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"



