import sys
sys.path.append('.')

import os

import pandas as pd

from experiments.performative_reasoning.analysis import run_analysis

models = [
  ('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'together', 'r1qwen'),
  ('Qwen/Qwen2.5-3B-Instruct', 'local', 'qwen3b'),
  ('meta-llama/Llama-3.2-3B', 'local', 'llama3b'),
  ('Qwen/Qwen2.5-7B-Instruct', 'together', 'qwen7b'),
  ('meta-llama/Meta-Llama-3.1-8B-Instruct', 'together', 'llama8b'),
]

results = pd.DataFrame(columns=[
  'Teacher',
  'Base',
  'Teacher Acc.',
  'Base Acc.',
  'Teacher->Base Acc.',
  'Difference vs. Base'
])

for base_model, base_strategy, base_name in models:
  for teacher_model, teacher_strategy, teacher_name in models:
    if base_model != teacher_model:

      if os.path.exists(f'results/swap_{teacher_name}_{base_name}.csv'):
        print(f'{teacher_name=} {base_name=}')

        teacher_acc, base_acc, teacher2base_acc = run_analysis(f'results/swap_{teacher_name}_{base_name}.csv')

        results.loc[len(results)] = [teacher_name, base_name, teacher_acc, base_acc, teacher2base_acc, teacher2base_acc - base_acc]
      else:
        print('skipping', f'results/swap_{teacher_name}_{base_name}.csv')

with open('context_swap_results.md', 'w') as f:
  f.write(results.to_markdown(index=False))