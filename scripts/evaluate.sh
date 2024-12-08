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
    python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${results_ckpt}/${tune_id}_$epoch.json --no_cache
done


# CUDA_VISIBLE_DEVICES=2 python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=prune_log/taylor/param_first/group_reduction/sum/0.5/pytorch_model.bin,config_pretrained=baffo32/decapoda-research-llama-7B-hf --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/taylor/param_first/group_reduction/sum/0.5/main.json --no_cache

# taylor_param_first_group_reduction_sum
# CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_first/group_reduction/sum prune_log/taylor/param_first/group_reduction/sum taylor/param_first/group_reduction/sum 200 400 600 800 1000 1200 1400

# taylor/param_second/group_reduction/sum
# CUDA_VISIBLE_DEVICES=2 bash scripts/evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_second/group_reduction/sum prune_log/taylor/param_second/group_reduction/sum taylor/param_second/group_reduction/sum 200 400 600 800 1000 1200 1400

# taylor/param_mix/group_reduction/sum
# CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate.sh baffo32/decapoda-research-llama-7B-hf tune_log/taylor/param_mix/group_reduction/sum prune_log/taylor/param_mix/group_reduction/sum taylor/param_mix/group_reduction/sum 200 400 600 800 1000 1200 1400
