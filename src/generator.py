from llm_sdk.llm_sdk import Small_LLM_Model


def generate(model: Small_LLM_Model):
	while (True):

		prompt_tensor = model.encode("What is the sum of 2 and 4?")

		input_ids_list = prompt_tensor[0].tolist()

		logits = model.get_logits_from_input_ids(input_ids_list)
