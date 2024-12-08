#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., decapoda-research/llama-7b-hf
tune_ckpt_name=$2 
prune_ckpt=$3
results_ckpt=$4
epochs=("${@:5}")

for epoch in "${epochs[@]}"; 
do
    # cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
    # mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

    tune_id="${tune_ckpt_name##*/}"
    python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks piqa --device cuda:0 --output_path results/${results_ckpt}/${tune_id}_$epoch.json --no_cache
done


# taylor_param_first_group_reduction_sum_arc-easy
# CUDA_VISIBLE_DEVICES=2 bash scripts/piqa/evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_first/group_reduction/sum/piqa/train/piqa_B4_G16_R8_A16 prune_log/taylor/param_first/group_reduction/sum taylor/param_first/group_reduction/sum/piqa/train/piqa_B4_G16_R8_A16 200 400 600 800 1000 1200 1400 1600 1800 2000