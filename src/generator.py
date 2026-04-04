import json
import numpy as np
from typing import List, Dict
from llm_sdk.llm_sdk import Small_LLM_Model


class ConstrainedGenerator():

	def __init__(self, model: Small_LLM_Model):

		self.model = model
		self.vocab = self._load_vocab()
		# self.id_to_token = {v: k for k, v in self.vocab.items()}
		# self.lbrace_id = self.vocab.get("{") or self.vocab.get("Ġ{")
		# self.quote_id = self.vocab.get('"')
		# self.colon_id = self.vocab.get(":")

	def _load_vocab(self) -> Dict[str, int]:
		"load vocabulary mapping from SDK's tokenizer file."
		vocab_path = self.model.get_path_to_vocab_file()
		# with open(vocab_path, "r", encoding='utf-8') as v:
		# 	data = json.load(v)
		return vocab_path

	def get_constrained_logit(self, logits: List[float], allowed_id: int) -> int:

		mask = np.full(len(logits), float('-inf'))
		mask[allowed_id] = 0
		final_logits = np.array(logits) + mask

		return int(np.argmax(final_logits))

model = Small_LLM_Model()

c = ConstrainedGenerator(model)

vocab = c._load_vocab()
mask = c.get_constrained_logit([1,2,3,4,5], 3)

print(vocab)

