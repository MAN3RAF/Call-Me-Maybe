from llm_sdk.llm_sdk import Small_LLM_Model
from typing import List, Dict
import numpy as np

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


def test1(model: Small_LLM_Model, prompt: str):

	ids = model.encode(prompt).tolist()[0]

	run = 1

	while True:
		
		logits = model.get_logits_from_input_ids(ids)

		if 1 in run:
			wanted_ids = get_next_token(logits, [8822])

		elif 2 in run:
			wanted_ids = get_next_token(logits, [])


		# res = logits.index(max(logits))

		# print(res)

		ids.append(wanted_ids)

		print(model.decode(ids))


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


import numpy as np
from typing import List
import json

# --- IMPORT YOUR MODEL HERE ---
# Adjust this import to match the name of your python file!
# from your_model_file import Small_LLM_Model 


def get_next_token(logits: List[float], allowed_ids: List[int]) -> int:
    """Forces the LLM to pick only from allowed_ids using -inf."""
    tokens = np.array(logits, dtype=float) + (-np.inf)

    for token_id in allowed_ids:
        tokens[token_id] = 0

    constrained_logits = np.array(logits, dtype=float) + tokens
    return int(np.argmax(constrained_logits))


def test(model, prompt: str, allowed_functions: List[str], vocab: Dict):

    allowed_paths = [model.encode(func).tolist()[0] for func in allowed_functions]
    print(allowed_paths)

    # Pre-compute all token IDs that represent numbers (for parameters)
    number_token_ids = [
        token_id for text, token_id in vocab.items() 
        if text.strip().isdigit() or text.strip() == "."
    ]
    print(number_token_ids)

    # ---------------------------------------------------------

    # 1. "WE DO": Force the start of the JSON
    json_start = f'{{\n  "prompt": "{prompt}",\n  "name": "'
    ids = model.encode(json_start).tolist()[0]
    
    print("--- Starting Generation ---")
    
    # 2. "LLM DOES": Choose the function name dynamically
    generated_func_ids = []
    
    while True:
        logits = model.get_logits_from_input_ids(ids)
        step = len(generated_func_ids)
        
        allowed_ids = []
        for path in allowed_paths:
            # If the path matches what we've generated so far...
            if path[:step] == generated_func_ids:
                if step < len(path):
                    allowed_ids.append(path[step])
                    
        # Force the LLM to pick from our calculated allowed_ids
        chosen_id = get_next_token(logits, allowed_ids)
        
        generated_func_ids.append(chosen_id)
        ids.append(chosen_id)
        
        # STOPPING CONDITION
        if generated_func_ids in allowed_paths:
            break

    # 3. "WE DO": Force the transition to parameters
    # Hardcoding "a" for testing purposes
    param_transition = '",\n  "parameters": {\n    "a": '
    ids += model.encode(param_transition).tolist()[0]

    # 4. "LLM DOES": Pick the number for parameter "a"
    logits = model.get_logits_from_input_ids(ids)
    chosen_number_id = get_next_token(logits, number_token_ids)
    ids.append(chosen_number_id)

    # 5. "WE DO": Close the JSON properly
    json_end = '\n  }\n}'
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
        "fn_add_numbers", 
        "fn_multiply_numbers", 
        "fn_reverse_string"
    ]
    
    # 4. Run the test!
    prompt = "What is the sum of 6 and 7?"
    test(model, prompt, allowed_funcs, vocab)



