from llm_sdk.llm_sdk import Small_LLM_Model
import sys
from typing import List, Dict, Any
import json
from src.func_utils import get_func_parameters, get_funcs, get_next_token, get_system_prompt, get_prompts, get_funcs_names


def get_number_token_ids(model: Small_LLM_Model ,type: str) -> List[int]:

    if type == "integer":
        
        allowed = [",", "}", "-", "0", "1", "2", "3", "4",
                   "5", "6", "7", "8", "9"]

        token_ids = [model.encode(a).tolist()[0][0] for a in allowed]

    if type == "number":
    
        allowed = [",", "}", "-", ".", "0", "1", "2", "3", "4",
                   "5", "6", "7", "8", "9"]

        token_ids = [model.encode(a).tolist()[0][0] for a in allowed]

    if type == "string":
        
        return []

    if type == "boolean":
         
        allowed = ['true', 'false']

        token_ids = [model.encode(a).tolist()[0][0] for a in allowed]

    return token_ids


def get_number_value(model: Small_LLM_Model, ids: Any, number_token_ids: List):
     
    while True:

        logits = model.get_logits_from_input_ids(ids)
        value_of_param_as_id = get_next_token(logits, number_token_ids)

        if model.decode([value_of_param_as_id]) == "," or model.decode([value_of_param_as_id]) == "}":
            break

        ids.append(value_of_param_as_id)


def get_string_value(model: Small_LLM_Model, ids: Any, number_token_ids: List):

    while True:

        logits = model.get_logits_from_input_ids(ids)
        value_of_param_as_id = get_next_token(logits, number_token_ids)

        token_text = model.decode([value_of_param_as_id])

        if '"' in token_text:
            ids += model.encode('"').tolist()[0]
            break

        ids.append(value_of_param_as_id)


def get_boolean_value(model: Small_LLM_Model, ids: Any, number_token_ids: List):

    logits = model.get_logits_from_input_ids(ids)
    value_of_param_as_id = get_next_token(logits, number_token_ids)

    ids.append(value_of_param_as_id)


def build_json(model: Small_LLM_Model, ids: Any, params: Dict):

    last_key = list(params.keys())[-1]

    for param_name, param_info in params.items():

        number_token_ids = get_number_token_ids(model, param_info['type'])

        if param_info['type'] == 'number' or param_info['type'] == 'integer':
            ids += model.encode(f'"{param_name}": ').tolist()[0]
            get_number_value(model, ids, number_token_ids)

        elif param_info['type'] == "string":
            ids += model.encode(f'"{param_name}": "').tolist()[0]
            get_string_value(model, ids, number_token_ids)

        elif param_info['type'] == "boolean":
            ids += model.encode(f'"{param_name}": ').tolist()[0]
            get_boolean_value(model, ids, number_token_ids)

        if param_name != last_key:
            ids += model.encode(", ").tolist()[0]
            
    ids += model.encode("}").tolist()[0]
[
        "fn_greet", 
        "fn_add_numbers",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_substitute_string_with_regex",
        'fn_answer_yes_no',
    ]

def json_generator(model: Small_LLM_Model, prompt: str, allowed_functions_names: List[str], funcs: List):

    allowed_paths = [model.encode(func).tolist()[0] for func in allowed_functions_names]

    # Pre-compute all token IDs that represent numbers (for parameters)

    # 1. "WE DO": Force the start of the JSON:

    sys_prompt = get_system_prompt(prompt, funcs)

    json_start = f'{{"prompt": "{prompt}", "name": "'
    
    json_start = sys_prompt + "\n\n" + json_start
    
    ids = model.encode(json_start).tolist()[0]

    print("--- Starting Generation ---")

    # 2. "LLM DOES": Choose the function name dynamically:

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

        # Force the LLM to pick from our calculated allowed_ids:

        chosen_id = get_next_token(logits, allowed_ids)

        generated_func_ids.append(chosen_id)
        ids.append(chosen_id)

        # STOPPING CONDITION:

        if generated_func_ids in allowed_paths:
            break

    # 3. "WE DO": Force the transition to parameters:

    params = get_func_parameters(model.decode(generated_func_ids), funcs)

    param_transition = f'", "parameters": {{'
    ids += model.encode(param_transition).tolist()[0]

    # 4. "LLM DOES": Pick the number for parameter "a":

    build_json(model, ids, params)

    # 5. "WE DO": Close the JSON properly:

    json_end = '}'
    ids += model.encode(json_end).tolist()[0]

    # --- FINAL OUTPUT ---

    final_text = model.decode(ids)

    final_text  = '{"prompt":' + final_text.split('{"prompt":')[1]

    print("\nFINAL JSON OUTPUT:\n")
    print(final_text)

def load_vocab(model: Small_LLM_Model) -> Dict[str, int]:
	"load vocabulary mapping from SDK's tokenizer file."
	vocab_path = model.get_path_to_vocab_file()
	with open(vocab_path, "r", encoding='utf-8') as v:
		data: Dict[str, int] = json.load(v)
	return data


def main():

    model = Small_LLM_Model() 

    parse = sys.argv

    try:
        if not parse[1] == '--functions_definition':
            raise ValueError("[Error] wrong '--functions_definition' format!")
        if not parse[3] == '--input':
            raise ValueError("[Error] wrong '--input' format!")
        if not parse[5] == '--output':
            raise ValueError("[Error] wrong '--output' format!")

        funcs_path = parse[2]
        prompts_path = parse[4]
        output_path = parse[6]

        if not '.json' in output_path:
            raise ValueError("[Error] wrong 'json' format!")

        with open(funcs_path, "r") as f:
            funcs = json.load(f)

        with open(prompts_path, "r") as p:
            prompts = json.load(p)

    except Exception as e:
        print(f"[Error] {str(e).split("]")[1].strip()}")

    allowed_funcs_names = get_funcs_names(funcs_path)

    for prompt in prompts:
        json_generator(model, prompt, allowed_funcs_names, funcs)


main()
