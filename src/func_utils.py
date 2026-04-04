from typing import List
import json
import numpy as np

def get_next_token(logits: List[float], allowed_ids: List[int]) -> int:
	"""Forces the LLM to pick only from allowed_ids using -inf."""
	
	if allowed_ids: 

		tokens = np.array(logits, dtype=float) + (-np.inf)

		for token_id in allowed_ids:
			tokens[token_id] = 0

		constrained_logits = np.array(logits, dtype=float) + tokens

		return int(np.argmax(constrained_logits))

	return logits.index(max(logits))


def get_func_parameters(func_name: str, funcs_list: List) -> List:

	for i in range(len(funcs_list)):

		if funcs_list[i]["name"] in func_name:
			par_dict = funcs_list[i]["parameters"]

			params = list(par_dict.keys())

	return params


def get_funcs(path: str) -> List:

	with open(path, "r") as f:
		data = json.load(f)

	
	return data


def get_prompts(path: str) -> List:

	with open(path, "r") as f:
		data = json.load(f)

	print(data)

