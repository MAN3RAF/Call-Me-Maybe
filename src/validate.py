from pydantic import BaseModel, ValidationError
from typing import Dict, List, Any
import json


class ParameterInfo(BaseModel):
    type: str


class FunctionValidator(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: Dict[str, str]


class PromptValidator(BaseModel):
    prompt: str


def parse(location: str) -> Any:
    """
    Parses a JSON file from the given location.

    Args:
        location: The path to the JSON file.

    Returns:
        The parsed JSON data as a Python object, or None if an error occurs.
    """
    try:
        with open(location, "r", encoding='utf-8') as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}")
        return None


def func_validate(location: str) -> List[Any]:
    """
    Validates a list of function definitions from a JSON file.

    Each function is validated against the FunctionValidator model. Invalid
    functions are skipped.

    Args:
        location: The path to the JSON file
        containing the function definitions.

    Returns:
        A list of valid function definitions as dictionaries.
    """
    data = parse(location)
    if data is None:
        return []

    valid_funcs = []
    for f in data:
        try:
            # Using **f automatically maps dict keys to class attributes
            f_v = FunctionValidator(**f)
            valid_funcs.append(f_v.model_dump())
        except (ValidationError, TypeError):
            return []
    return valid_funcs


def prompt_validate(location: str) -> List[Any]:
    """
    Validates a list of prompts from a JSON file.

    Each prompt is validated against the PromptValidator model. Invalid
    prompts are skipped.

    Args:
        location: The path to the JSON file containing the prompts.

    Returns:
        A list of valid prompts as dictionaries.
    """
    data = parse(location)
    if data is None:
        return []

    valid_prompts = []
    for p in data:
        try:
            p_v = PromptValidator(**p)
            valid_prompts.append(p_v.model_dump())
        except (ValidationError, TypeError):
            return []
    return valid_prompts
