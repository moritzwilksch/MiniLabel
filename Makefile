install:
	pip install -r requirements.txt

format:
	isort .
	black .

frontend:
	uvicorn main:app --reload --app-dir src/frontend