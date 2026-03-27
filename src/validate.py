from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class ParameterInfo(BaseModel):
	type: str


class FunctionDefinition(BaseModel):
	name: str
	description: str
	parameers: Dict[str, ParameterInfo]
	returns: Dict[str, str]

