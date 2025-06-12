import sys
sys.path.append('.')

from omegaconf import OmegaConf
import litellm
from experiments.performative_reasoning.inference_strategies import TogetherAPI, LocalUnslothModel, APIModel
from datasets import load_dataset
import pandas as pd
import argparse

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from experiments.performative_reasoning.utils import get_git_revision_short_hash, extract_hash_answer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

config = OmegaConf.load(f'configs/{args.config}.yaml')

if config.base_inference_strategy == 'together':
  base_model = TogetherAPI(config.base_model, max_tokens=config.max_generated_tokens, temperature=config.temperature)
elif config.base_inference_strategy == 'local':
  base_model = LocalUnslothModel(config.base_model, max_tokens=config.max_generated_tokens, temperature=config.temperature)
else:
  raise NotImplementedError

if config.teacher_inference_strategy == 'together':
  teacher_model = TogetherAPI(config.teacher_model, max_tokens=config.max_generated_tokens, temperature=config.temperature)
elif config.teacher_inference_strategy == 'local':
  teacher_model = LocalUnslothModel(config.teacher_model, max_tokens=config.max_generated_tokens, temperature=config.temperature)
else:
  raise NotImplementedError

data = load_dataset('openai/gsm8k', 'main')['train']

results = pd.DataFrame(columns=[
  'Question',
  'Base Completion',
  'Teacher Completion',
  'Modified Base Completion',
  'Correct Answer'
])

contexts = []

for i in range(config.n_samples):
  question = data[i]['question']
  answer = data[i]['answer']
  ground_truth_answer = extract_hash_answer(answer)

  results.loc[i] = [
    question,
    str(),
    str(),
    str(),
    ground_truth_answer
  ]

  context = [
    {'role': 'system', 'content': config.system_prompt},
    {'role': 'user', 'content': question},
  ]

  contexts.append(context)

base_completions = base_model.generate_completions(contexts)

teacher_completions = teacher_model.generate_completions(contexts)

modified_contexts = []

if config.context_modifier == 'swap':
  # get raw reasoning traces

  gpt_model = APIModel('gpt-4o-mini', max_tokens=config.max_generated_tokens * 2, temperature=0.4)

  reword_contexts = [
    [
      {'role': 'system', 'content': 'Given a reasoning trace provided by the user, sanitize the reasoning trace so that:\n1. It does not contain any XML tags like <reasoning>, </reasoning>, <answer>, or </answer>.\n2. It does not mention the final answer at all, whether in the reasoning trace or in the answer XML tags.\n\nMake whatever edits are necessary to satisfy conditions 1. and 2., and leave the rest of the reasoning trace unchanged otherwise. Make sure the final answer is removed from the reasoning trace too, if it is mentioned at the last step. Output the modified reasoning trace verbatim.'},
      {'role': 'user', 'content': teacher_completion},
    ]
    for teacher_completion in teacher_completions
  ]

  drop_last_sentence = lambda s: s[:s[:s.rfind('.')].rfind('.') + 1]
  cleaned_traces = [drop_last_sentence(trace) for trace in gpt_model.generate_completions(reword_contexts)]

  teacher_traces = []
  for context, trace in zip(
    contexts, cleaned_traces
  ):
    modified_context = context + [{
      'role': 'assistant', 
      'content': f'<reasoning>{trace}</reasoning>'
    }]
    modified_contexts.append(modified_context)

modified_completions = base_model.generate_completions(modified_contexts, continuation=True)

for i in range(config.n_samples):
  results.iloc[:, 1] = base_completions
  results.iloc[:, 2] = teacher_completions
  results.iloc[:, 3] = modified_completions

results.to_csv(f'results/{args.config}.csv')