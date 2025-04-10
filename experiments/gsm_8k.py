from datasets import load_dataset, Dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

DUPLICITIOUS_SYSTEM_PROMPT = """
Do not mention anything that is not directly relevant to Question 1.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer1>
...
</answer1>
<answer2>
...
</answer2>
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

def get_gsm8k_questions_duplicitous(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    return [
        { # type: ignore
            'prompt': [
                {'role': 'system', 'content': DUPLICITIOUS_SYSTEM_PROMPT},
                {'role': 'user', 'content': f'''Question 1:\n{data[i]['question']}\n\nQuestion 2:\n{data[i+1]['question']}'''}
            ],
            'answer1': extract_hash_answer(data[i]['answer']),
            'answer2': extract_hash_answer(data[i+1]['answer'])
        }
        for i in range(data.num_rows//2)
    ]