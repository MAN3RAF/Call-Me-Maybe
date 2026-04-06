import sys

def main() -> None:

    parse = sys.argv
    try:
        if not parse[1] == '--functions_definition':
            raise ValueError("[Error] wrong '--functions_definition' format!")
        if not parse[3] == '--input':
            raise ValueError("[Error] wrong '--input' format!")
        if not parse[5] == '--output':
            raise ValueError("[Error] wrong '--output' format!")


        funcs_path = parse[2]
        prompts = parse[4]
        output_path = parse[6]

        if not '.json' in output_path:
            raise ValueError("[Error] wrong 'json' format!")

    
        with open(funcs_path, "r") as f:
            pass

        with open(prompts, "r") as f:
            pass

    except Exception as e:
        print(f"[Error] {str(e).split("]")[1].strip()}")


    # print(f"\n{funcs_path}\n\n{prompts}\n\n{output_path}\n")


if __name__ == "__main__":

    main()
