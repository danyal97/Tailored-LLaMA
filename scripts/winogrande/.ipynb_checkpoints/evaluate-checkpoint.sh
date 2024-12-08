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
    python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks winogrande --device cuda:0 --output_path results/${results_ckpt}/${tune_id}_$epoch.json --no_cache
done


# taylor_param_first_group_reduction_sum_winogrande
# CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_first/group_reduction/sum/0.5/winogrande/train/B4_R8_A16_G16 prune_log/taylor/param_first/group_reduction/sum/0.5 taylor/param_first/group_reduction/sum/0.5/winogrande/train/B4_R8_A16_G16 4600



# python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=prune_log/taylor/param_first/group_reduction/sum/pytorch_model.bin,peft=tune_log/taylor/param_first/group_reduction/sum/winogrande/custom_method,config_pretrained=baffo32/decapoda-research-llama-7B-hf --tasks winogrande --device cuda:0 --output_path results/taylor/param_first/group_reduction/sum/winogrande/custom_method/result.json --no_cache

# CUDA_VISIBLE_DEVICES=3 bash scripts/winogrande_evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_first/group_reduction/sum/winogrande/custom_method prune_log/taylor/param_first/group_reduction/sum taylor/param_first/group_reduction/sum/winogrande/custom_method 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000


# CUDA_VISIBLE_DEVICES=2 bash scripts/winogrande_evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_first/group_reduction/sum/winogrande/custom_method-batch-size-4 prune_log/taylor/param_first/group_reduction/sum taylor/param_first/group_reduction/sum/winogrande/custom_method-batch-size-4 200 400 600