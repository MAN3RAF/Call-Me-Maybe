from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict, List
import json
from pathlib import Path


class PromptValidator(BaseModel):
	prompt: Dict[str, str]


class FunctionValidator(BaseModel):
	name: str
	description: str
	parameters: Dict[str, Dict[str, str]]
	returns: Dict[str, str]


class Validator:

	def validate():

		try:
			with open("data/input/function_calling_tests.json", "r") as pr:
				with open("data/input/functions_definition.json", "r") as fun:
					# prompts = json.load(pr)
					# for p in prompts:
					# 	p_v = PromptValidator(prompt=p)
					# 	print(p_v)

					funcs = json.load(fun)
					for f in funcs:
						f_v = FunctionValidator(name=f['name'], description=f['description'], parameters=f['parameters'], returns=f['returns'])
						print(f_v)

		except FileNotFoundError:
			print("ERROR: No json file was found!")
		except ValidationError:
			print("Validation Error!")
		except Exception:
			print("ERROR: wrong json format!")
		# print(prompts)

v = Validator
v.validate()