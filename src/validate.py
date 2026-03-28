from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict, List
import json
from pathlib import Path
import sys


class PromptValidator(BaseModel):
	prompt: Dict[str, str]


class FunctionValidator(BaseModel):
	name: str
	description: str
	parameters: Dict[str, Dict[str, str]]
	returns: Dict[str, str]



def prompt_validate(location):

	try:
		with open(location, "r") as pr:
			prompts = json.load(pr)
			for p in prompts:
				try:
					p_v = PromptValidator(prompt=p)
					print("Good")
				except ValidationError:
					print("VAlidation Error!")
				except Exception:
					print("Bad")

	except FileNotFoundError:
		print("ERROR: No json file was found!")
	except Exception:
		print("ERROR: wrong json format!")


def func_validate(location):


	try:
		with open(location, "r") as fun:

			funcs = json.load(fun)
			for f in funcs:
				try:
					f_v = FunctionValidator(name=f['name'], description=f['description'], parameters=f['parameters'], returns=f['returns'])
					print("Good")
				except ValidationError:
					print("VAlidation Error!")
				except Exception:
					print("Bad")

	except FileNotFoundError:
		print("bad")
	except Exception:
		print("bad ")
