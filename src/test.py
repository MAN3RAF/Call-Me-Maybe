import sys
import os

def main() -> None:

    try:

        prompts:str = None
        output_path:str = None
        funcs_path:str = None


        for i in range(len(sys.argv)):

            if sys.argv[i] == '--input' and i + 1 < len(sys.argv):
                prompts = sys.argv[i + 1] 
            elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
                output_path = sys.argv[i + 1] 
            elif sys.argv[i] == '--functions_definition' and i + 1 < len(sys.argv):
                funcs_path = sys.argv[i + 1]

        if len(sys.argv) > 7:
            raise ValueError("[Error] wrong command format, dont add garbage to the command!")

        if not prompts:
            raise ValueError("[Error] '--input' wrong format!")
        if not funcs_path:
            raise ValueError("[Error] '--functions_definition' wrong format!")
        if not output_path:
            raise ValueError("[Error] '--output' wrong format!")

        if not os.path.exists(prompts):
            raise ValueError(f"[Error] The file '{prompts}' does not exist!")
        if not os.path.exists(funcs_path):
            raise ValueError(f"[Error] The file '{funcs_path}' does not exist!")

        if not output_path.endswith('.json'):
            raise ValueError(f"[Error] '{output_path}' wrong 'json' format!")


    except ValueError as e:
        print(e)

if __name__ == "__main__":

    main()
