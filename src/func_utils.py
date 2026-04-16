from typing import Dict, List, Any
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


def get_func_parameters(func_name: str,
                        funcs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Retrieves the parameters of a specific function
        from a list of functions.

    Args:
        func_name: The name of the function to find.
        funcs_list: A list of dictionaries, where each dictionary represents a
                    function and contains its name and parameters.

    Returns:
        A dictionary containing the parameters of the specified function.
        Returns an empty dictionary if the function is not found.
    """

    params: Dict[str, Any] = {}

    for i in range(len(funcs_list)):

        if funcs_list[i]["name"] == func_name:

            params = funcs_list[i]["parameters"]

    return params


def get_funcs_names(data: List[Dict[str, Any]]) -> List[str]:
    """Extracts the names of all functions from a list of function definitions.

    Args:
        data: A list of dictionaries, where each dictionary represents a
              function and contains a 'name' key.

    Returns:
        A list of strings, where each string is the name of a function.
    """

    funcs_names = []

    for d in data:
        funcs_names.append(d['name'])

    return funcs_names


def get_system_prompt(prompt: str, funcs: List[Dict[str, Any]]) -> str:
    """Build a structured prompt with the user request and function defs.

    Args:
        prompt: The original natural language prompt.
        functions: List of available function definitions.

    Returns:
        A formatted prompt string suitable for the LLM.
    """

    system_prompt = ("You are a smart AI. You must choose the correct function"
                     " from the list below based on the user's prompt.\n\n\n")

    system_prompt += f"User request: {prompt}\n"

    system_prompt += "\nAvailable functions:\n"

    for func in funcs:
        func_name = func['name']
        func_desc = func['description']
        func_param = func['parameters']

        system_prompt += f"\n{func_name}\n"

        for param_name, param_details in func_param.items():
            param_type = param_details['type']
            system_prompt += f"({param_name}: {param_type})\n"

        system_prompt += f"{func_desc}."

    system_prompt += ("\n\nRespond with a JSON object with keys 'name' and "
                      "'parameters'.\n\n")

    return system_prompt
