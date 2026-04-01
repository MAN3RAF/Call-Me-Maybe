from llm_sdk.llm_sdk import Small_LLM_Model


model = Small_LLM_Model()

prompt = "Whats the sum of 2 and 4?"

while True:
	ids = model.encode(prompt)

	logits = model.get_logits_from_input_ids(ids.tolist()[0])

	print(f'{model.decode(logits.index(max(logits)))}')
