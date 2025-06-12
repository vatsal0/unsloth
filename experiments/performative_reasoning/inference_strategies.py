from abc import ABC, abstractmethod
import random
import time

import asyncio
import litellm
import openai
from tqdm import tqdm
from unsloth import FastLanguageModel
from vllm import SamplingParams

litellm.suppress_debug_info = True

together_endpoints = {
  'meta-llama/Meta-Llama-3.1-8B-Instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
  'Qwen/Qwen2.5-7B-Instruct': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
}

class InferenceStrategy(ABC):

  @abstractmethod
  def generate_completions(self, messages, continuation=False):
    pass

class APIModel(InferenceStrategy):
  def __init__(self, model_name, max_tokens, temperature):
    self.model_name = model_name
    self.max_tokens = max_tokens
    self.temperature = temperature
    self.retries = 5

  async def generate_completion(self, messages):
    success = False
    for i in range(self.retries):
      try:
        response = await litellm.acompletion(
          model=self.model_name,
          max_tokens=self.max_tokens,
          temperature=self.temperature,
          messages=messages
        )
        success = True
        break
      except (litellm.ContentPolicyViolationError, openai.APIError) as e:
        if isinstance(e, openai.APIError) and not litellm._should_retry(e.status_code):
            raise
        time.sleep(random.random() * 2**i)
    if not success:
        raise RuntimeError(f"Failed after {self.retries} retries.")

    return response.choices[0].message.content

  async def gather_completions(self, contexts):
    tasks = [self.generate_completion(context) for context in tqdm(contexts)]
    results = await asyncio.gather(*tasks)
    return results

  def generate_completions(self, contexts, continuation=False):
    return asyncio.run(self.gather_completions(contexts))

class TogetherAPI(APIModel):
  def __init__(self, model_name, max_tokens, temperature):
    if model_name in together_endpoints.keys(): 
      model_name = together_endpoints[model_name]

    self.model_name = f'together_ai/{model_name}'
    self.max_tokens = max_tokens
    self.temperature = temperature
    self.retries = 5


class LocalUnslothModel(InferenceStrategy):
  # TODO
  # Init unsloth model
  # Call unsloth model.generate
  def __init__(self, model_name, max_tokens, temperature):
    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
      model_name = model_name,
      max_seq_length = max_tokens,
      load_in_4bit = True,
      fast_inference = True,
      max_lora_rank = 64,
      gpu_memory_utilization = 0.4,
    )

    seed = time.time_ns() & 8191
    self.sampling_params = SamplingParams(
        temperature = temperature,
        top_p = 0.95,
        max_tokens = max_tokens,
        seed = seed
    )

  def generate_completions(self, messages, continuation=False):
    inputs = [
      self.tokenizer.apply_chat_template(
        message, 
        tokenize = False, 
        add_generation_prompt = False,
        continue_final_message = continuation,
      )
      for message in messages]
    return [
      output.outputs[0].text 
      for output in self.model.fast_generate(inputs, sampling_params=self.sampling_params)
    ]

class OpenRouterAPI(InferenceStrategy):
  # TODO

  def generate_completion(self, messages):
    raise NotImplementedError
