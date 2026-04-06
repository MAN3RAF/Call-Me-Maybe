from src.main import main


try:
	main()

except ValueError as e:
	print(e)