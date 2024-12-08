import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import re
import torch
import random
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    TrainerCallback
)
from LLMPruner.peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Set CUDA device

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the dataset
dataset = load_dataset("hellaswag")

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def _process_doc(doc,i):
    ctx = doc["ctx_a"][i] + " " + doc["ctx_b"][i].capitalize()
    out_doc = {
        "query": preprocess(doc["activity_label"][i] + ": " + ctx),
        "choices": [preprocess(ending) for ending in doc["endings"][i]],
        "gold": int(doc["label"][i]),
    }
    return out_doc

def formatting_prompts_func(examples):
    output_text = []
    for i, gold in enumerate(examples["label"]):
        doc = _process_doc(examples,i)
        choices = doc["choices"]
        response = doc["gold"]
        input_text = doc["query"]
        instruction_block = f"Question: {input_text}\n"
        instruction_block += f"Answer: {choices[response]}"
        
        # print(instruction_block)
        
        output_text.append(instruction_block)
    return output_text

def create_model_and_tokenizer():
    pruned_dict = torch.load("prune_log/taylor/param_first/group_reduction/sum/0.5/pytorch_model.bin", map_location='cpu')
    model = prepare_model_for_int8_training(pruned_dict['model'])
    tokenizer = pruned_dict['tokenizer']

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() 
    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False

OUTPUT_DIR = "tune_log/taylor/param_first/group_reduction/sum/0.5/hellaswag/train/hellaswag_B4_G16_R8_A16"
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    optim="adamw_torch",
    num_train_epochs=30,
    logging_steps=10,
    logging_first_step=True,
    learning_rate=1e-4,
    # weight_decay=0.,
    fp16=True,
    warmup_steps=100,
    group_by_length=False,
    # warmup_ratio=0.03,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_strategy="steps",
    ddp_find_unused_parameters=None,
    lr_scheduler_type="cosine",
    report_to="wandb",
    save_safetensors=True,
    load_best_model_at_end=True,
    seed=42,
    run_name=OUTPUT_DIR.split('/')[-1],
)

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint-{state.global_step}")
            model.save_pretrained(checkpoint_path)

callbacks = [PeftSavingCallback()]



# print(tokenizer.encode("Question: A satellite image can help scientists locate the area where two plates have diverged by showing the existence of\nAnswer: rift valleys.", add_special_tokens=False)[2:])

response_template = "\nAnswer:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
# print(response_template_ids)
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    formatting_func=formatting_prompts_func,
    packing=False,
    max_seq_length=256,
    tokenizer=tokenizer,
    data_collator=collator,
    callbacks=callbacks
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
