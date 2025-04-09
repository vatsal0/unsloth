import pandas as pd
from experiments.rewards import extract_xml_answer
from experiments.gsm_8k import extract_hash_answer
import numpy as np

accuracies = []
for file in [
  # 'results_672.csv',
  # 'results_2386.csv',
  # 'results_4667.csv',
  # 'results_5821.csv',
  # 'results_6274.csv',
  # 'results_6967.csv',
  # 'results_7004.csv',
  # 'results_7484.csv',
  # 'results_7812.csv',
  # 'results_8103.csv',
  'results_330.csv',
  'results_372.csv',
  'results_1434.csv',
  'results_2300.csv',
  'results_5797.csv',
  'results_6768.csv',
  'results_7722.csv',
]:
  results = pd.read_csv(file)
  results = results[~results['GRPO reasoning trace'].isna()]
  ground_truth = results['Correct answer'].apply(extract_hash_answer)
  answers = results.iloc[:, 3:9].map(extract_xml_answer)
  for i in range(6):
    accuracies.append((answers.iloc[:, i] == ground_truth).mean())

accuracies = np.array(accuracies).reshape(-1, 6)
