*This project has been created as part of the 42 curriculum by lsebar

# Call-Me-Maybe

## Description

**Call-Me-Maybe** is a project focused on implementing **Function Calling** for Large Language Models (LLMs), with a specific emphasis on reliability in small models (Qwen3-0.6B). The goal is to translate natural language requests (e.g., "What is the sum of 40 and 2?") into structured, machine-executable JSON objects. 

To overcome the unreliability of small models in generating structured output, this project utilizes **constrained decoding**. This technique guides the model token-by-token, ensuring that the output strictly adheres to a predefined JSON schema without relying on prompting alone.

## Instructions

### Prerequisites
- Python 3.10+
- `uv` package manager (recommended)

### Installation
Install the project dependencies using the provided `Makefile`:

```bash
make install
```

Alternatively, using `uv` directly:
```bash
uv sync
```

### Execution
Run the main script using the following command:

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

By default, the program reads from `data/input/` and writes to `data/output/`.

### Other Commands
- `make lint`: Run flake8 and mypy checks.
- `make clean`: Remove temporary files and caches.
- `make debug`: Run the script in debug mode with `pdb`.

## Algorithm Explanation

The project implements **Constrained Decoding** via **Logit Masking**. Instead of allowing the LLM to generate any token from its vocabulary, we intercept the generation process at each step:

1.  **State Machine Guidance**: The system maintains the state of the JSON structure (e.g., currently inside the "name" field, or expecting a "number" value).
2.  **Logit Masking**: For each generation step, we identify the set of "valid" tokens that maintain both JSON syntax and schema compliance. All other tokens in the model's vocabulary have their logits set to negative infinity (`-inf`).
3.  **Prefix Matching**: When selecting a function name, the model is constrained to only pick tokens that form a valid path toward one of the available function names defined in `functions_definition.json`.
4.  **Type Enforcement**: For parameters, we restrict the model to specific character sets:
    -   `number`/`integer`: Only digits, signs, and decimal points.
    -   `boolean`: Only `true` or `false`.
    -   `string`: Any characters until a closing quote is generated.

This ensures that the model **cannot** produce invalid JSON or hallucinate non-existent functions.

## Design Decisions

-   **Pydantic for Validation**: We use Pydantic to ensure that the input files (`functions_definition.json` and `function_calling_tests.json`) are correctly formatted before processing, providing clear error messages if they aren't.
-   **Hybrid Injection**: We combine static text injection (e.g., `{"prompt": "...", "name": "`) with dynamic LLM generation for names and values. This reduces overhead and guarantees structural integrity.
-   **Small Model Optimization**: By offloading structural responsibility to the decoder, we enable a 0.6B parameter model to achieve production-grade reliability (99%+ valid JSON).

## Performance Analysis

-   **Accuracy**: Achieves near 100% structural accuracy. Function selection accuracy depends on the LLM's understanding of the prompt but is significantly enhanced by removing invalid options.
-   **Reliability**: Every output is guaranteed to be parseable by standard JSON libraries.
-   **Speed**: Processing remains fast enough to handle multiple prompts in under a minute, as the logit masking overhead is minimal compared to the model's forward pass.

## Challenges Faced

-   **Tokenizer Nuances**: Handling how the model represents numbers (e.g., tokenizing "123.45" differently than "123") required careful prefix matching and character-level constraints.
-   **Space/Control Tokens**: Dealing with leading spaces (e.g., token `Ġ` in BPE) was critical for matching function names accurately.
-   **Error Handling**: Managing edge cases like missing files or malformed input JSON without crashing the entire pipeline.

## Testing Strategy

The implementation was validated using:
1.  **Unit Tests**: Verifying individual components like `get_next_token` and `encode_decode`.
2.  **Integration Tests**: Running the full pipeline against `function_calling_tests.json` and comparing outputs against the function schemas.
3.  **Edge Case Testing**: Providing ambiguous prompts or functions with complex types to test the limits of the constrained decoder.

## Example Usage

**Input Prompt**: "Reverse the string 'hello'"
**Expected Output**:
```json
{
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
        "s": "hello"
    }
}
```

## Resources

-   **Qwen3 Documentation**: Details on the model architecture and vocabulary.
-   **JSON Schema**: Standard definitions for structured data representation.
-   **Python Typing/Mypy**: Documentation for ensuring code robustness.
-   **AI Usage**: AI tools were used to assist in writing PEP-257 docstrings, optimizing logit masking logic for performance, and structuring this documentation. All generated code was manually reviewed and tested to ensure compliance with the 42 coding standards.
