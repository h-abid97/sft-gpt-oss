"""
LoRA Fine-Tuning GPT-OSS 20B with Unsloth + TRL
----------------------------------------------
This script shows an end-to-end workflow for fine-tuning the GPT-OSS 20B model
using LoRA adapters for parameter-efficient training.

Steps:
1) Load GPT-OSS 20B in 4-bit precision (reduces VRAM usage)
2) Attach LoRA adapters (train only ~1% of parameters)
3) Load and format the Multilingual-Thinking dataset
4) Fine-tune with Hugging Face TRL's SFTTrainer
5) Test inference before and after training
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Prevents TorchDynamo compile errors in some environments

from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from datasets import load_dataset
from transformers import TextStreamer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

# ============================================================
# 1) Load GPT-OSS base model in 4-bit precision
# ============================================================
print("\n\n=== 1) Loading GPT-OSS 20B in 4-bit ===")
# 4-bit loading drastically reduces memory requirements
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    dtype=None,                  # Auto-detect best dtype
    max_seq_length=4096,          # Set context length
    load_in_4bit=True,            # Memory optimization
    full_finetuning=False         # We'll use LoRA instead
)

# ============================================================
# 2) Attach LoRA adapters for efficient fine-tuning
# ============================================================
print("\n\n\n\n=== 2) Adding LoRA Adapters ===")
# LoRA allows training only small "adapter" layers, saving time & GPU memory
model = FastLanguageModel.get_peft_model(
    model,
    r=8,                          # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # Layers to adapt
    lora_alpha=16,                # Scaling factor
    lora_dropout=0,               # Dropout for LoRA layers (0 for speed)
    bias="none",                  # Don't train bias terms
    use_gradient_checkpointing="unsloth",  # Save memory with longer context
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)
model.config.use_cache = False    # Needed when using gradient checkpointing

# ============================================================
# 3) Quick test inference before training
# ============================================================

# ------------------------------------------------------------------------------
# Reasoning Effort (GPT-OSS):
# Controls the trade-off between reasoning depth and speed by adjusting
# how many tokens the model uses to "think."
#
# Levels:
#   * Low:
#       Optimized for tasks that need very fast responses and don't require
#       complex, multi-step reasoning.
#   * Medium:
#       A balance between performance and speed.
#   * High:
#       Provides the strongest reasoning performance for tasks that require it,
#       though this results in higher latency.
# ------------------------------------------------------------------------------

print("\n\n\n\n=== 3) Quick Pre-Training Inference ===")
messages = [{"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low"  # Set reasoning effort to low, medium or high
).to(model.device)

_ = model.generate(**inputs, max_new_tokens=128, streamer=TextStreamer(tokenizer))

# ============================================================
# 4) Load & format the training dataset
# ============================================================

# ------------------------------------------------------------------------------
# The `HuggingFaceH4/Multilingual-Thinking` dataset will be used as our example.
# This dataset, available on Hugging Face, contains reasoning chain-of-thought
# examples derived from user questions, translated from English into four other
# languages. The purpose of using this dataset is to enable the model to learn 
# and develop reasoning capabilities in these four distinct languages.
# ------------------------------------------------------------------------------

print("\n\n\n\n=== 4) Loading & Formatting Dataset ===")
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
dataset = standardize_sharegpt(dataset) # Standardize to ShareGPT-style format so GPT-OSS chat template works

# Convert conversation format to plain text prompts
def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in examples["messages"]
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print("Sample formatted text:\n", dataset[0]["text"], "...")

# ============================================================
# 5) Fine-tune with TRL's SFTTrainer
# ============================================================
print("\n\n\n\n=== 5) Starting LoRA Fine-Tuning ===")
# Data collator pads sequences & prepares labels for causal LM training
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,  # Ignore padding in loss computation
    pad_to_multiple_of=8,     # Tensor cores optimization
    return_tensors="pt"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    packing=True,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # Effective batch = 8
        warmup_steps=5,
        max_steps=60,                    # Short demo run (increase for real training)
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",               # Memory-efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none"
    )
)

trainer.train()

# ============================================================
# 6) Test inference after training
# ============================================================
print("\n\n\n\n=== 6) Running Post-Training Inference ===")
messages = [
    {"role": "system", "content": "reasoning language: French\n\nYou are a helpful assistant that can solve mathematical problems."},
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."}
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="medium"  # Try medium reasoning effort this time
).to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
_ = model.generate(**inputs, max_new_tokens=1024, streamer=streamer)