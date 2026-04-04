from llm_sdk.llm_sdk import Small_LLM_Model
from typing import List, Dict, Any
import numpy as np
import json
from src.func_utils import get_func_parameters, get_funcs, get_next_token


def get_next_token1(logits: List, allowed_ids: List) -> int:

	tokens = np.array(logits) + (- np.inf)

	for id in allowed_ids:
		tokens[id] = 0

	constrained_logits = np.array(logits) + tokens
	
	return int(np.argmax(constrained_logits))


def generate_text(llm: Small_LLM_Model, prompt: str, max_tokens: int = 50) -> str:

    input_tensor = llm.encode(prompt)
    input_ids_list = input_tensor.tolist()[0]

    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(input_ids_list)

        max_score_id = int(np.argmax(logits))

        input_ids_list.append(max_score_id)

    return llm.decode(input_ids_list)


def main():
	model = Small_LLM_Model()

	prompt = "What is the sum of 2 and 4?"
	allowed_ids = [1, 0]
	# i = 100
	# while i:

	# 	ids = model.encode(prompt)

	# 	logits = model.get_logits_from_input_ids(ids.tolist()[0])

	# 	# print(get_next_token(logits, allowed_ids))
	# 	# print(model.decode(logits.index(max(logits))))
	# 	print(logits)

	# 	# prompt += model.decode(logits.index(max(logits)))
	# 	# i -= 1

	# print(generate_text(model, prompt, 50))

	# vocab = model.get_path_to_vocab_file()

	# print(vocab)

	test(model, f'{{\n  "prompt": "{prompt}",\n  "name": "')

def get_number_token_ids(model: Small_LLM_Model ,type: str) -> List[int]:

    if type == "number" or type == "integer":
        
        allowed = [",", "}", "-", ".", "0", "1", "2", "3", "4",
                   "5", "6", "7", "8", "9"]

        token_ids = [model.encode(a).tolist()[0][0] for a in allowed]

    if type == "string":
        
        return []

    return token_ids



def build_number_json(ids: Any, params: List, number_token_ids: List):
     
    for i in params:

        print(params)


        ids += model.encode(f'"{i}": ').tolist()[0]

        while True:

            logits = model.get_logits_from_input_ids(ids)
            value_of_param_as_id = get_next_token(logits, number_token_ids)

            # print(model.decode(value_of_param_as_id))

            if model.decode([value_of_param_as_id]) == "," or model.decode([value_of_param_as_id]) == "}":
                break

            ids.append(value_of_param_as_id)

        if i != params[-1]:
            ids += model.encode(", ").tolist()[0]
        else:
            ids += model.encode("}").tolist()[0]
            break

    return ids

def build_string_json(ids: Any, params: List, number_token_ids: List):
     
    for i in params:

        print(params)


        ids += model.encode(f'"{i}": "').tolist()[0]

        while True:

            logits = model.get_logits_from_input_ids(ids)
            value_of_param_as_id = get_next_token(logits, number_token_ids)

            # print(model.decode(value_of_param_as_id))
            
            token_text = model.decode([value_of_param_as_id])

            if '"' in token_text:
                ids += model.encode('"').tolist()[0]
                break

            ids.append(value_of_param_as_id)

        if i != params[-1]:
            ids += model.encode('", ').tolist()[0]
        else:
            ids += model.encode("}").tolist()[0]
            break

    return ids

def test(model, prompt: str, allowed_functions: List[str], type: str):

    allowed_paths = [model.encode(func).tolist()[0] for func in allowed_functions]

    print([model.decode(path) for path in allowed_paths])

    # Pre-compute all token IDs that represent numbers (for parameters)


    number_token_ids = get_number_token_ids(model, type)


    # 1. "WE DO": Force the start of the JSON:

    json_start = f'{{"prompt": "{prompt}", "name": "'
    ids = model.encode(json_start).tolist()[0]

    print("--- Starting Generation ---")

    # 2. "LLM DOES": Choose the function name dynamically:

    generated_func_ids = []

    while True:
        logits = model.get_logits_from_input_ids(ids)
        step = len(generated_func_ids)

        allowed_ids = []

        # print(allowed_ids)

        for path in allowed_paths:
            # If the path matches what we've generated so far...
            # print(model.decode(path))
            if path[:step] == generated_func_ids:
                # print(model.decode(generated_func_ids))
                if step < len(path):
                    # print(step)
                    allowed_ids.append(path[step])

        # Force the LLM to pick from our calculated allowed_ids:

        chosen_id = get_next_token(logits, allowed_ids)

        # print(model.decode(chosen_id))

        generated_func_ids.append(chosen_id)
        ids.append(chosen_id)

        # STOPPING CONDITION:

        # print(model.decode(generated_func_ids))
        # print(model.decode(allowed_paths[0]))

        if generated_func_ids in allowed_paths:
            # print(model.decode(generated_func_ids))
            # print("Broke")
            break
    # 3. "WE DO": Force the transition to parameters:

    ##### Have to be edited #####
    funcs = get_funcs("data/input/functions_definition.json")

    params = get_func_parameters(model.decode(generated_func_ids), funcs)

    param_transition = f'", "parameters": {{'
    ids += model.encode(param_transition).tolist()[0]

    # 4. "LLM DOES": Pick the number for parameter "a":

    # logits = model.get_logits_from_input_ids(ids)
    # value_of_param_as_id = get_next_token(logits, number_token_ids)

    if type == "number" or type == "integer":
        ids = build_number_json(ids, params, number_token_ids)
    elif type == "string":
         ids = build_string_json(ids, params, number_token_ids)

    # 5. "WE DO": Close the JSON properly:

    json_end = '}'
    ids += model.encode(json_end).tolist()[0]

    # --- FINAL OUTPUT ---

    final_text = model.decode(ids)
    print("\nFINAL JSON OUTPUT:\n")
    print(final_text)

def load_vocab(model: Small_LLM_Model) -> Dict[str, int]:
	"load vocabulary mapping from SDK's tokenizer file."
	vocab_path = model.get_path_to_vocab_file()
	with open(vocab_path, "r", encoding='utf-8') as v:
		data = json.load(v)
	return data





if __name__ == "__main__":
    # 1. Initialize your model (Adjust to your actual class name)
    model = Small_LLM_Model() 

    # 2. Get the vocabulary dictionary from your model
    # (e.g., model.vocab, model.tokenizer.get_vocab(), etc.)
    vocab = load_vocab(model) 

    # 3. Define the allowed functions for this test
    allowed_funcs = [
        "fn_greet", 
        "fn_add_numbers",
        "fn_reverse_string",
        # "fn_get_square_root",
        "fn_substitute_string_with_regex",
    ]

    # 4. Run the test!
    prompt = "Great John"
    test(model, prompt, allowed_funcs, "string")



