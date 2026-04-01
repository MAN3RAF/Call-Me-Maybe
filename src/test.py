from llm_sdk.llm_sdk import Small_LLM_Model
from typing import List, Any
import numpy as np

def get_next_token(logits: List, allowed_ids: List) -> int:

	tokens = np.array(logits) + -float("inf")

	for id in allowed_ids:
		tokens[id] = 0

	constrained_logits = np.array(logits) + tokens
	
	print(constrained_logits)





def main():
	model = Small_LLM_Model()

	prompt = "What is the sum of 2 and 4?"

	while True:

		ids = model.encode(prompt)

		logits = model.get_logits_from_input_ids(ids.tolist()[0])

		print(model.decode(logits.index(max(logits))))

		prompt += model.decode(logits.index(max(logits)))


get_next_token([1,2,3,4,5,6], [2,3])
