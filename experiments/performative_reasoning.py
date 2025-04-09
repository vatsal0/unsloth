from unsloth import FastLanguageModel
from vllm import SamplingParams
from time import time_ns
from datasets import load_dataset
import pandas as pd

from experiments.gsm_8k import extract_hash_answer
from experiments.rewards import extract_xml_answer

GRPO_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

BASE_SYSTEM_PROMPT = """
Output only your final answer as a single number and nothing else. DO NOT EXPLAIN YOUR ANSWER. Respond in the following format:
<answer>
...
</answer>
"""

REASONING_END = "</reasoning>"

NUM_EXAMPLES = 100

dataset = load_dataset('openai/gsm8k', 'main')['test'][:NUM_EXAMPLES]

base_model_path = "Qwen/Qwen2.5-3B-Instruct"
base_model_path = "meta-llama/Llama-3.2-3B"
grpo_model_path = "/lus/eagle/projects/DemocAI/vatsalb/grpo_runs/lr 1.0e-05 grad clip 0.05 constant_with_warmup/lr 1.0e-05 grad clip 0.05 constant_with_warmup/checkpoint-750"

load_model = lambda path: FastLanguageModel.from_pretrained(
    model_name = path,
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 64,
    gpu_memory_utilization = 0.4,
)

base_model, tokenizer = load_model(base_model_path)
grpo_model, tokenizer = load_model(grpo_model_path)

for _ in range(10):
  seed = time_ns() & 8191
  sampling_params = SamplingParams(
      temperature = 0.6,
      top_p = 0.95,
      max_tokens = 1024,
      seed = seed
  )

  inputs_1 = [
    tokenizer.apply_chat_template([
      {"role" : "system", "content" : GRPO_SYSTEM_PROMPT},
      {"role" : "user", "content" : question},
    ], tokenize = False, add_generation_prompt = True)
    for question in dataset["question"]
  ]
  outputs_1 = grpo_model.fast_generate(inputs_1, sampling_params=sampling_params)

  reasoning_traces = []
  skip_indices = []

  for i, output in enumerate(outputs_1):
    output_text = output.outputs[0].text
    if REASONING_END in output_text:
      reasoning_traces.append(output_text[:output_text.find(REASONING_END) + len(REASONING_END)])
    else:
      reasoning_traces.append('')

  from experiments.rewards import extract_xml_answer
  extract_xml_answer(outputs_1[0].outputs[0].text)

  inputs_2 = [
    tokenizer.apply_chat_template([
      {"role" : "system", "content" : BASE_SYSTEM_PROMPT},
      {"role" : "user", "content" : question},
    ], tokenize = False, add_generation_prompt = True)
    for question in dataset["question"]
  ]
  outputs_2 = base_model.fast_generate(inputs_2, sampling_params=sampling_params)

  inputs_3 = [
    tokenizer.apply_chat_template([
      {"role" : "system", "content" : BASE_SYSTEM_PROMPT},
      {"role" : "user", "content" : question},
    ], tokenize = False, add_generation_prompt = True) + '<answer>' # MAKE SURE YOU ADD THAT IN BEFORE PARSING
    for question in dataset["question"]
  ]
  outputs_3 = base_model.fast_generate(inputs_3, sampling_params=sampling_params)

  inputs_4 = [
    tokenizer.apply_chat_template([
      {"role" : "system", "content" : BASE_SYSTEM_PROMPT},
      {"role" : "user", "content" : question},
    ], tokenize = False, add_generation_prompt = True) + reasoning_trace + '\n' + '<answer>'
    for question, reasoning_trace in zip(dataset["question"], reasoning_traces)
  ]
  outputs_4 = base_model.fast_generate(inputs_4, sampling_params=sampling_params)

  inputs_5 = [
    tokenizer.apply_chat_template([
      {"role" : "system", "content" : BASE_SYSTEM_PROMPT},
      {"role" : "user", "content" : question + '\n' + reasoning_trace},
    ], tokenize = False, add_generation_prompt = True) + '<answer>'
    for question, reasoning_trace in zip(dataset["question"], reasoning_traces)
  ]
  outputs_5 = base_model.fast_generate(inputs_5, sampling_params=sampling_params)

  inputs_6 = [
    tokenizer.apply_chat_template([
      {"role" : "system", "content" : BASE_SYSTEM_PROMPT},
      {"role" : "user", "content" : reasoning_trace + '\n' + question},
    ], tokenize = False, add_generation_prompt = True) + '<answer>'
    for question, reasoning_trace in zip(dataset["question"], reasoning_traces)
  ]
  outputs_6 = base_model.fast_generate(inputs_6, sampling_params=sampling_params)

  results = pd.DataFrame(columns=[
    'Question',
    'GRPO reasoning trace',

    'Output, GRPO', 
    'Output, Base',
    'Output, Base w/ answer forcing',
    'Output, Base w/ trace in answer',
    'Output, Base w/ trace after question',
    'Output, Base w/ trace before question',

    'Correct answer'
  ])

  for i in range(NUM_EXAMPLES):
    results.loc[i] = [
      dataset["question"][i],
      reasoning_traces[i],
      outputs_1[i].outputs[0].text,
      outputs_2[i].outputs[0].text,
      '<answer>' + outputs_3[i].outputs[0].text,
      '<answer>' + outputs_4[i].outputs[0].text,
      '<answer>' + outputs_5[i].outputs[0].text,
      '<answer>' + outputs_6[i].outputs[0].text,
      dataset["answer"][i],
    ]

  results.to_csv(f'results_{seed}.csv')
