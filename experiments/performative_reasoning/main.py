import sys
sys.path.append('.')

from omegaconf import OmegaConf
import litellm
from experiments.performative_reasoning.inference_strategies import TogetherAPI
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

if config.inference_strategy == 'together':
  base_model = TogetherAPI(config.base_model, max_tokens=config.max_generated_tokens, temperature=config.temperature)
  teacher_model = TogetherAPI(config.teacher_model, max_tokens=config.max_generated_tokens, temperature=config.temperature)
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
for context, base_completion, teacher_completion in zip(
  contexts, base_completions, teacher_completions
):
  if config.context_modifier == 'swap':
    modified_context = context + [{
      'role': 'assistant', 
      'content': teacher_completion[:teacher_completion.find('</reasoning>') + len('</reasoning>')]
    }]
    modified_contexts.append(modified_context)

modified_completions = base_model.generate_completions(modified_contexts)

for i in range(config.n_samples):
  results.iloc[:, 1] = base_completions
  results.iloc[:, 2] = teacher_completions
  results.iloc[:, 3] = modified_completions

results.to_csv(f'results/{args.config}_{get_git_revision_short_hash()}.csv')
