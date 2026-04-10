from src.main import generate_json

try:
    generate_json()
except ValueError as e:
    print(e)
