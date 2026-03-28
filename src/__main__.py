import argparse
import src.validate as validate

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--functions_definition", 
                        default="data/input/functions_definition.json",
                        help="Path to function definitions")
    
    parser.add_argument("--input", 
                        default="data/input/function_calling_tests.json",
                        help="Path to input prompts")
    
    parser.add_argument("--output", 
                        default="data/output/function_calling_results.json",
                        help="Path to save results")

    args = parser.parse_args()

    validate.func_validate(args.functions_definition)
    validate.prompt_validate(args.input)

if __name__ == "__main__":
    main()