from abc import ABC, abstractmethod
from tqdm import tqdm
import time
import random

import asyncio
import litellm
import openai

litellm.suppress_debug_info = True

class InferenceStrategy(ABC):

  @abstractmethod
  def generate_completion(self, messages):
    pass

class TogetherAPI(InferenceStrategy):
  def __init__(self, model_name, max_tokens, temperature):
    self.model_name = f'together_ai/{model_name}'
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

  def generate_completions(self, contexts):
    return asyncio.run(self.gather_completions(contexts))


class LocalUnslothModel(InferenceStrategy):
  # TODO
  # Init unsloth model
  # Call unsloth model.generate

  def generate_completion(self, messages):
    raise NotImplementedError

class OpenRouterAPI(InferenceStrategy):
  # TODO

  def generate_completion(self, messages):
    raise NotImplementedError
