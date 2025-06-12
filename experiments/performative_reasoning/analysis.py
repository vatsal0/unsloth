import sys
sys.path.append('.')

import argparse

import pandas as pd
from experiments.performative_reasoning.utils import extract_xml_answer

def run_analysis(file_path, print_stats=False):
  data = pd.read_csv(file_path).fillna('')

  for col in ['Base Completion', 'Teacher Completion', 'Modified Base Completion']:
    extracted_answers = data[col].map(extract_xml_answer)

  for i, row in data.iterrows():
    correct_answer = str(row['Correct Answer'])

    for col in ['Base Completion', 'Teacher Completion', 'Modified Base Completion']:
      extracted_answer = extract_xml_answer(row[col])
      correct = correct_answer in extracted_answer.replace('$', '').replace('.', '').split()
      data.loc[i, col.replace('Completion', 'Correct')] = correct

  if print_stats:
    print(f'Teacher Accuracy: {data['Teacher Correct'].sum() / len(data):0.2f}')
    print(f'Base Accuracy: {data['Base Correct'].sum() / len(data):0.2f}')
    print(f'Modified Base Accuracy: {data['Modified Base Correct'].sum() / len(data):0.2f}')

    print('Failure Cases:')
    for i, row in data[data['Teacher Correct'].astype(bool) & ~data['Modified Base Correct'].astype(bool)].iterrows():
      print(i, 'Correct Answer:', row['Correct Answer'])
      print('Correct Teacher:')
      print(row['Teacher Completion'])
      print('\nIncorrect Modified Base:')
      print(row['Modified Base Completion'])
      print('\n\n')

  return data['Teacher Correct'].sum() / len(data), data['Base Correct'].sum() / len(data), data['Modified Base Correct'].sum() / len(data)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_path', type=str, required=True)
  args = parser.parse_args()

  run_analysis(args.file_path, print_stats=True)