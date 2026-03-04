.PHONY: install run test docker-build docker-run clean

install:
	pip install -r requirements.txt

run:
	uvicorn app.api.server:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

docker-build:
	docker build -t vector-retail-agent .

docker-run:
	docker run -p 8000:8000 --env-file .env vector-retail-agent

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
