from llm_sdk.llm_sdk import Small_LLM_Model
from typing import List
import numpy as np

def get_next_token(logits: List, allowed_ids: List) -> int:

	tokens = np.array(logits) + -float("inf")

	for id in allowed_ids:
		tokens[id] = 0

	constrained_logits = np.array(logits) + tokens
	
	return int(np.argmax(constrained_logits))





def main():
	model = Small_LLM_Model()

	prompt = "Greet Shrek!"
	allowed_ids = [1, 0]
	i = 100
	while i:

		ids = model.encode(prompt)

		logits = model.get_logits_from_input_ids(ids.tolist()[0])

		# print(get_next_token(logits, allowed_ids))
		# print(model.decode(logits.index(max(logits))))

		prompt += model.decode(logits.index(max(logits)))
		i -= 1



main()




