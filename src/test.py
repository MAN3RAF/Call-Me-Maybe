import argparse
import src.validate as validate
from src.func_utils import get_func_parameters
from src.__main__ import main

def test() -> None:

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

    func_list = validate.func_validate(args.functions_definition)
    prompt_list = validate.prompt_validate(args.input)

    g = get_func_parameters("fn_substitute_string_with_regex", func_list)

    print(g)


if __name__ == "__main__":

    main()