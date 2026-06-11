from llm_sdk.llm_sdk import Small_LLM_Model
from encode_decode import encode, decode
import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import json
from src.func_utils import (get_func_parameters, get_next_token,
                            get_system_prompt, get_funcs_names)
from src.validate import func_validate, prompt_validate


def get_number_token_ids(model: Small_LLM_Model, type: str) -> List[int]:
    """
    Retrieves a list of token IDs corresponding to allowed characters for a
    specific data type.

    This function is used to constrain the language model's output to valid
    characters for numbers (integer and float), booleans, and strings.

    Args:
        model: The language model instance, used to encode characters into
               token IDs.
        type: The data type for which to get the allowed token IDs.
              Supported types are "integer", "number", "string", and "boolean".

    Returns:
        A list of integer token IDs. For "string", it returns an empty list
        as any character is allowed within a string.
    """

    token_ids: List[int] = []

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


def get_number_value(model: Small_LLM_Model, ids: Any,
                     number_token_ids: List[int]) -> None:
    """
    Generates a numeric value (integer or float) token by token, constrained
    by a set of allowed token IDs.

    The generation stops when a comma or a closing brace is predicted,
    indicating the end of the number.

    Args:
        model: The language model instance.
        ids: A list of token IDs representing the input sequence so far.
             This list is modified in-place.
        number_token_ids: A list of token IDs that are allowed for
                          numeric values.
    """

    while True:

        logits = model.get_logits_from_input_ids(ids)
        value_of_param_as_id = get_next_token(logits, number_token_ids)

        if (model.decode([value_of_param_as_id]) == "," or
                model.decode([value_of_param_as_id]) == "}"):
            break

        ids.append(value_of_param_as_id)


def get_string_value(model: Small_LLM_Model, ids: Any,
                     number_token_ids: List[int]) -> None:
    """
    Generates a string value token by token.

    The generation stops when a double quote is predicted, indicating the end
    of the string.

    Args:
        model: The language model instance.
        ids: A list of token IDs representing the input sequence so far.
             This list is modified in-place.
        number_token_ids: An unused parameter, present for consistency with
                          other value generation functions.
    """

    while True:

        logits = model.get_logits_from_input_ids(ids)
        value_of_param_as_id = get_next_token(logits, number_token_ids)

        token_text = model.decode([value_of_param_as_id])

        if '"' in token_text:
            ids += model.encode('"').tolist()[0]
            break

        ids.append(value_of_param_as_id)


def get_boolean_value(model: Small_LLM_Model,
                      ids: Any,
                      number_token_ids: List[int]) -> None:
    """
    Generates a boolean value ('true' or 'false') by predicting the next token.

    Args:
        model: The language model instance.
        ids: A list of token IDs representing the input sequence so far.
             This list is modified in-place.
        number_token_ids: A list of token IDs allowed for boolean values.
    """

    logits = model.get_logits_from_input_ids(ids)
    value_of_param_as_id = get_next_token(logits, number_token_ids)

    ids.append(value_of_param_as_id)


def build_json(model: Small_LLM_Model, ids: Any,
               params: Dict[str, Any]) -> None:
    """
    Constructs the parameters part of the JSON object by generating values for
    each parameter based on its type.

    Args:
        model: The language model instance.
        ids: A list of token IDs representing the input sequence so far.
             This list is modified in-place.
        params: A dictionary describing the parameters of the function,
                including their names and types.
    """

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


def json_generator(model: Small_LLM_Model, prompt: str,
                   allowed_functions_names: List[str],
                   funcs: List[Dict[str, Any]]) -> Any:
    """
    Generates a JSON object representing a function call based on a natural
    language prompt.

    This function orchestrates the process of:
    1. Creating a system prompt.
    2. Forcing the model to choose a function name from the allowed list.
    3. Generating the parameters for the chosen function.
    4. Assembling and returning the final JSON object.

    Args:
        model: The language model instance.
        prompt: The natural language user prompt.
        allowed_functions_names: A list of function names that the model is
                                 allowed to choose from.
        funcs: A list of dictionaries defining the available functions.

    Returns:
        A dictionary representing the generated JSON object, which includes the
        original prompt, the chosen function name, and its parameters.
    """

    allowed_paths = [model.encode(func).tolist()[0]
                     for func in allowed_functions_names]

    # Pre-compute all token IDs that represent numbers (for parameters)

    # 1. "WE DO": Force the start of the JSON:

    sys_prompt = get_system_prompt(prompt, funcs)

    safe_prompt = json.dumps(prompt)

    # Inject it WITHOUT adding your own quotes around it!
    json_start = f'{{"prompt": {safe_prompt}, "name": "'

    json_start = sys_prompt + "\n\n" + json_start

    ids = model.encode(json_start).tolist()[0]

    # 2. "LLM DOES": Choose the function name dynamically:

    generated_func_ids: List[int] = []

    print("\n== generating function's name ==")

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

    print(f"-->[{model.decode(generated_func_ids)}]")

    # 3. "WE DO": Force the transition to parameters:

    print("\n== Generating Parameters ==")

    params = get_func_parameters(model.decode(generated_func_ids), funcs)

    param_transition = '", "parameters": {'
    ids += model.encode(param_transition).tolist()[0]

    # 4. "LLM DOES": Pick the number for parameter "a":

    build_json(model, ids, params)

    # 5. "WE DO": Close the JSON properly:

    json_end = '}'
    ids += model.encode(json_end).tolist()[0]

    # --- FINAL OUTPUT ---

    final_text = model.decode(ids)

    final_text = '{"prompt":' + final_text.split('{"prompt":')[1]

    clean_text = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', final_text)

    final_text = json.loads(clean_text)

    for param_name, param_info in params.items():

        if param_info['type'] == 'number':
            final_text['parameters'][param_name] = float(
                final_text['parameters'][param_name]
                )

        elif param_info['type'] == 'integer':
            final_text['parameters'][param_name] = int(
                final_text['parameters'][param_name]
                )
        print(f"-->[{param_name}: {final_text['parameters'][param_name]}]")

    print("\n== JSON Output ==")
    print(final_text)

    return final_text


def generate_json() -> None:
    """
    Main function to generate function call JSON objects
    from a file of prompts.

    It reads prompts from an input file, generates a JSON for each, and writes
    the results to an output file. File paths can be specified
    via command-line arguments.

    Command-line arguments:
        --input: Path to the input JSON file containing prompts.
                 Defaults to "data/input/function_calling_tests.json".
        --output: Path to the output JSON file.
                  Defaults to "data/output/function_calls.json".
        --functions_definition: Path to the JSON file defining the functions.
                            Defaults to "data/input/functions_definition.json".

    Raises:
        ValueError: If command-line arguments are incorrect, or if specified
                    files do not exist or have the wrong format.
    """

    prompts_path: str = "data/input/function_calling_tests.json"
    output_path_str: str = "data/output/function_calling_results.json"
    funcs_path: str = "data/input/functions_definition.json"

    print("=== Parsing the Input ===")

    for i in range(len(sys.argv)):

        if sys.argv[i] == '--input' and i + 1 < len(sys.argv):
            prompts_path = sys.argv[i + 1]
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_path_str = sys.argv[i + 1]
        elif sys.argv[i] == '--functions_definition' and i + 1 < len(sys.argv):
            funcs_path = sys.argv[i + 1]

    if len(sys.argv) > 7:
        raise ValueError(
            "[Error] wrong command format, dont add garbage to the command!")

    if not os.path.exists(prompts_path):
        raise ValueError(f"[Error] The file '{prompts_path}' does not exist!")
    if not os.path.exists(funcs_path):
        raise ValueError(f"[Error] The file '{funcs_path}' does not exist!")

    if not output_path_str.endswith('.json'):
        raise ValueError(f"[Error] '{output_path_str}' wrong 'json' format!")

    prompts: List[Dict[str, Any]] = prompt_validate(prompts_path)

    funcs: List[Dict[str, Any]] = func_validate(funcs_path)

    if not funcs:
        raise ValueError("[Error] No valid functions found or invalid file.")

    if not prompts:
        print("[Warning] No valid prompts found or invalid file.",
              file=sys.stderr)

    model = Small_LLM_Model()

    allowed_funcs_names: List[str] = get_funcs_names(funcs)

    result: List[Any] = []

    output_path = Path(output_path_str)

    print('-->[Input Clear!]')

    print("\n=== Starting the JSON generation ===")

    for prompt in prompts:
        prompt_text: str = prompt['prompt']

        result.append(json_generator(model, prompt_text,
                                     allowed_funcs_names, funcs))

    print("\n=== Ending generation ===")
    print("-->[Outputing everything in a JSON file]")

    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
