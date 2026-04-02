from llm_sdk.llm_sdk import Small_LLM_Model
from typing import List
import numpy as np

def get_next_token(logits: List, allowed_ids: List) -> int:

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


def test(model: Small_LLM_Model, prompt: str):

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


main()



