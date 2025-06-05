# Makefile - 로컬 테스트 및 개발을 위한 명령어들

.PHONY: help install test test-ci lint format clean docker-test

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests locally"
	@echo "  test-ci     - Run tests in CI mode (without MLflow)"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean temporary files"
	@echo "  docker-test - Run tests in Docker"

install:
	pip install -r requirements.txt
	pip install -r test_requirements.txt

test:
	ENABLE_MLFLOW=true pytest tests/ -v --cov=src

test-ci:
	ENABLE_MLFLOW=false CI=true pytest tests/ -v --cov=src --cov-report=xml

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

docker-test:
	docker build -f Dockerfile.test -t weather-predictor-test .
	docker run --rm weather-predictor-test

# 개발 모드로 실행 (MLflow 활성화)
run-dev:
	ENABLE_MLFLOW=true python main.py run_all

# CI 모드로 실행 (MLflow 비활성화)
run-ci:
	ENABLE_MLFLOW=false python main.py run_all