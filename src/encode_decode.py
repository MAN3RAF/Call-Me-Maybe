from typing import List
import json
from llm_sdk.llm_sdk import Small_LLM_Model

def encode(model: Small_LLM_Model, text: str) -> List[int]:
    """Encodes text using a longest-match algorithm."""
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    text = text.replace(" ", "Ġ").replace("\n", "Ċ")

    res = []
    j = 0
    while j < len(text):
        match_found = False

        for i in range(len(text), j, -1):
            substring = text[j:i]
            if substring in vocab:
                res.append(vocab[substring])
                j = i
                match_found = True
                break

        if not match_found:
            j += 1

    return res

def decode(model: Small_LLM_Model, ids: List[int]) -> str:
    """Decodes token IDs back to a string."""
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    inverse_vocab = {v: k for k, v in vocab.items()}

    tokens = []
    for i in ids:
        if i in inverse_vocab:
            tokens.append(inverse_vocab[i])

    res = "".join(tokens)
    return res.replace("Ġ", " ").replace("Ċ", "\n")
