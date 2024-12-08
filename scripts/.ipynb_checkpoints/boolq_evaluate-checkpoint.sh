#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., decapoda-research/llama-7b-hf
tune_ckpt_name=$2 
prune_ckpt=$3
results_ckpt=$4
epochs=("${@:5}")

for epoch in "${epochs[@]}"; 
do
    cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
    mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

    tune_id="${tune_ckpt_name##*/}"
    python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks boolq --device cuda:0 --output_path results/${results_ckpt}/${tune_id}_$epoch.json --no_cache
done

# /home/danyal/LLM-FINAL/
# taylor_param_first_group_reduction_sum_boolq
# CUDA_VISIBLE_DEVICES=1 bash scripts/boolq_evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_first/group_reduction/sum/boolq/train/BOOLQ_B8_R8_A16_G16 prune_log/taylor/param_first/group_reduction/sum taylor/param_first/group_reduction/sum/boolq/train/BOOLQ_B8_R8_A16_G16 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4300 4400 4500 4600 4700 4800 4900 5000 5100 5200 5300 5400 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700



# python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=prune_log/taylor/param_first/group_reduction/sum/pytorch_model.bin,peft=tune_log/taylor/param_first/group_reduction/sum/boolq/custom_method,config_pretrained=baffo32/decapoda-research-llama-7B-hf --tasks boolq --device cuda:0 --output_path results/taylor/param_first/group_reduction/sum/boolq/custom_method/result.json --no_cache

# CUDA_VISIBLE_DEVICES=3 bash scripts/boolq_evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_first/group_reduction/sum/boolq/custom_method prune_log/taylor/param_first/group_reduction/sum taylor/param_first/group_reduction/sum/boolq/custom_method 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000


# CUDA_VISIBLE_DEVICES=2 bash scripts/boolq_evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_first/group_reduction/sum/boolq/custom_method-batch-size-4 prune_log/taylor/param_first/group_reduction/sum taylor/param_first/group_reduction/sum/boolq/custom_method-batch-size-4 200 400 600