import torch
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model() # Ensure it's initialized correctly

# 1. Encode your prompt
text = "What is the sum of 2 and 3?"
tokens = model.encode(text) 

# 2. Convert to a 2D Tensor: Shape (1, sequence_length)
# This is likely what was missing!
input_tensor = torch.tensor([tokens]) 

# 3. Get logits
logits = model.get_logits_from_input_ids(input_tensor)