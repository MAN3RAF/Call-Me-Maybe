from pydantic import BaseModel, ValidationError
from typing import Dict, List, Optional
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


def parse(location: str) -> List | None:
    try:
        with open(location, "r", encoding='utf-8') as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}")
        return None


def func_validate(location: str) -> List:
    data = parse(location)
    if data is None: return []

    valid_funcs = []
    for f in data:
        try:
            # Using **f automatically maps dict keys to class attributes
            f_v = FunctionValidator(**f)
            valid_funcs.append(f_v.model_dump())
        except (ValidationError, TypeError) as e:
            print(f"ERROR: Skipping invalid function entry.")
    return valid_funcs


def prompt_validate(location: str) -> List:
    data = parse(location)
    if data is None: return []

    valid_prompts = []
    for p in data:
        try:
            p_v = PromptValidator(**p)
            valid_prompts.append(p_v.model_dump())
        except (ValidationError, TypeError):
            print("ERROR: Skipping invalid prompt entry.")
    return valid_prompts
