import re

def extract_xml_answer(text: str) -> str:
    answer1 = text.split("<answer1>")[-1]
    answer1 = answer1.split("</answer1>")[0]
    answer2 = text.split("<answer2>")[-1]
    answer2 = answer2.split("</answer2>")[0]
    return answer1.strip(), answer2.strip()

# Reward functions
def get_words(sentence):
    return set(re.sub(r'[^\w\s]', '', sentence).lower().split())

def second_reasoning(q, response):
  q1 = re.search(r'Question 1:\n(.*)\n', q).group(1)
  q2 = re.search(r'Question 2:\n(.*)', q).group(1)
  q1_proper = set(re.findall(r'\b[A-Z][a-z]*\b', q1))
  q2_proper = set(re.findall(r'\b[A-Z][a-z]*\b', q2))
  return (q2_proper - q1_proper).intersection(get_words(response)) != set() 

def correctness_reward_func(prompts, completions, answer1, answer2, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer1[0]}, {answer2[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [-1.0 if second_reasoning(q, r_full) else (r[0] == a1)*0.5 + (r[1] == a2)*1.5 for r, a1, a2, r_full in zip(extracted_responses, answer1, answer2, responses)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.25 * r1.isdigit() + 0.25 * r2.isdigit() for r1, r2 in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer1>\n.*?\n</answer1>\n<answer2>\n.*?\n</answer2>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer1>.*?</answer>1\s*<answer2>.*?</answer2>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer1>\n") == 1:
        count += 0.0625
        count -= len(text.split("\n</answer1>\n")[-1])*0.001
    if text.count("\n</answer1>") == 1:
        count += 0.0625
        count -= (len(text.split("\n</answer1>")[-1]) - 1)*0.001
    if text.count("\n<answer2>\n") == 1:
        count += 0.0625
        count -= len(text.split("\n</answer2>\n")[-1])*0.001
    if text.count("\n</answer2>") == 1:
        count += 0.0625
        count -= (len(text.split("\n</answer2>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]