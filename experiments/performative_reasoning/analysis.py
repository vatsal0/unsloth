import sys
sys.path.append('.')

import argparse

import pandas as pd
from experiments.performative_reasoning.utils import extract_xml_answer

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, required=True)
args = parser.parse_args()

data = pd.read_csv(args.file_path).fillna('')

for col in ['Base Completion', 'Teacher Completion', 'Modified Base Completion']:
  extracted_answers = data[col].map(extract_xml_answer)

for i, row in data.iterrows():
  correct_answer = str(row['Correct Answer'])

  for col in ['Base Completion', 'Teacher Completion', 'Modified Base Completion']:
    extracted_answer = extract_xml_answer(row[col])
    correct = correct_answer in extracted_answer.replace('$', '').replace('.', '').split()
    data.loc[i, col.replace('Completion', 'Correct')] = correct

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
