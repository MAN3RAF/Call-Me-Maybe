from typing import Dict, List, Any
import json
from llm_sdk.llm_sdk import Small_LLM_Model

def encode():
	pass








def decode(model: Small_LLM_Model, ids: List[int] | Any) -> str:

	url = model.get_path_to_vocab_file()
	print(url)
	f = open(url, "r")
	vocab = json.load(f)
	# print(vocab)
	res = ""
	print(ids)
	for i in ids:
		if i in vocab.values():
			print("Here!")
			res += "".join([k for k, v in vocab.items() if v == i][0])
	res = res.replace("Ġ", " ")
	res = res.replace("Ċ", "\n")
	return res


model = Small_LLM_Model()

encoded = model.encode("Hello World!").tolist()[0]

print(decode(model, encoded))
