from typing import List, Dict
import json
import numpy as np

def get_next_token(logits: List[float], allowed_ids: List[int]) -> int:
	"""Forces the LLM to pick only from allowed_ids using -inf."""
	
	if not allowed_ids:
		return logits.index(max(logits))

	tokens = np.array(logits, dtype=float) + (-np.inf)

	for token_id in allowed_ids:
		tokens[token_id] = 0

	constrained_logits = np.array(logits, dtype=float) + tokens

	return int(np.argmax(constrained_logits))


def get_func_parameters(func_name: str, funcs_list: List[Dict]) -> Dict:

	params: List = []

	for i in range(len(funcs_list)):

		if funcs_list[i]["name"] == func_name:

			params = funcs_list[i]["parameters"]

	return params


def get_funcs(path: str) -> List:

	with open(path, "r") as f:
		data = json.load(f)

	
	return data


def get_prompts(path: str) -> List:

	with open(path, "r") as f:
		data = json.load(f)

	print(data)


def get_system_prompt(funcs: List) -> str:

	system_prompt = "You are a smart AI. You must choose the correct function from the list below based on the user's prompt.\n\nAvailable functions:\n"


	for func in funcs:
        
		func_name = func['name']
		
		func_desc = func['description']

		func_param = func['parameters']

		system_prompt += f"\n- {func_name}: {func_desc}\n  Parameters:\n"

		for param_name, param_details in func_param.items():

			param_type = param_details['type']

			system_prompt += f"  - {param_name}: ({param_type})\n"

	system_prompt += "\nExtract the correct function and parameters for this prompt into JSON:\n\n"

	return system_prompt



def get_func_type(func_name: str, funcs: List):
	
	for i in range(len(funcs)):

		if funcs[i]["name"] == func_name:
			par_dict = funcs[i]["parameters"]

			params = list(par_dict.values())
	
	print(params)





# get_func_parameters("fn_add_numbers", get_funcs("data/input/functions_definition.json"))



# get_func_type("fn_add_numbers", get_funcs("data/input/functions_definition.json"))