import os
from contextlib import chdir

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("-d", "--device_id", type=int, required=True)
parser.add_argument("--lora_rank", type=int, default=64)
parser.add_argument("--seq_length", type=int, default=1024)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--grad_clip", type=float, default=0.1)
parser.add_argument("--lr_schedule", type=str, default="constant_with_warmup")
parser.add_argument("--output_dir", type=str, default="testing")
args = parser.parse_args()

os.environ["WANDB_PROJECT"] = "cot_backdoor"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import GRPOConfig, GRPOTrainer

from experiments.gsm_8k import get_gsm8k_questions
from experiments.rewards import xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, int_reward_func, correctness_reward_func

dataset = get_gsm8k_questions()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model,
    max_seq_length = args.seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = args.lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = args.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = args.lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

training_args = GRPOConfig(
    # Params that will probably not change
    use_vllm = True,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    report_to = "wandb", # Can use Weights & Biases
    
    # Params that could sometimes change
    warmup_ratio = 0.1,
    max_prompt_length = 256,
    max_completion_length = 256,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1000,
    save_strategy = "steps",
    save_steps = 250,

    # Params that will frequently change
    learning_rate = args.learning_rate,
    max_grad_norm = args.grad_clip,
    lr_scheduler_type = args.lr_schedule, # originally cosine
    gradient_accumulation_steps = 1, 
    per_device_train_batch_size = 64, # needs to be a multiple of num_generations
    num_generations = 16, 
    output_dir = args.output_dir,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

os.makedirs(f'/lus/eagle/projects/DemocAI/vatsalb/grpo_runs/{args.output_dir}', exist_ok=True)
with chdir(f'/lus/eagle/projects/DemocAI/vatsalb/grpo_runs/{args.output_dir}'): # HACK there's a 'grpo_trainer_lora_model' config that's saved and reloaded 
# at every inference step so we want to keep each run in a separate folder
  trainer.train() # resume_from_checkpoint = True
  model.save_lora("grpo_saved_lora")