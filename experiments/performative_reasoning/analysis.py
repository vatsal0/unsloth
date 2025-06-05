import sys
sys.path.append('.')

import pandas as pd
from experiments.performative_reasoning.utils import extract_xml_answer

file_path = 'results/swap_0a70e49.csv'

data = pd.read_csv(file_path)

for col in ['Base Completion', 'Teacher Completion', 'Modified Base Completion']:
  extracted_answers = data[col].map(extract_xml_answer)

for i, row in data.iterrows():
  correct_answer = str(row['Correct Answer'])

  for col in ['Base Completion', 'Teacher Completion', 'Modified Base Completion']:
    extracted_answer = extract_xml_answer(row[col])
    correct = correct_answer in extracted_answer.replace('$', '').replace('.', '').split(' ')
    data.loc[i, col.replace('Completion', 'Correct')] = correct

print(f'Base Accuracy: {data['Base Correct'].sum() / len(data):0.2f}')
print(f'Teacher Accuracy: {data['Teacher Correct'].sum() / len(data):0.2f}')
print(f'Modified Base Accuracy: {data['Modified Base Correct'].sum() / len(data):0.2f}')

print('Failure Cases:')
for i, row in data[data['Teacher Correct'].astype(bool) & ~data['Modified Base Correct'].astype(bool)].iterrows():
  print('Correct Teacher:')
  print(row['Teacher Completion'])
  print('Incorrect Modified Base:')
  print(row['Modified Base Completion'])
  print('\n\n')
