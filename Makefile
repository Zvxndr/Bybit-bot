# Bybit Trading Bot Makefile
# Common development and deployment tasks

.PHONY: help install dev test lint format clean docker-build docker-dev docker-prod

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  dev          - Run in development mode"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-dev   - Run development environment with Docker"
	@echo "  docker-prod  - Run production environment with Docker"

# Python environment setup
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Development
dev:
	python -m src.bot.main --debug --paper-trade

dev-dashboard:
	python -m src.bot.main --dashboard-only --debug

# Testing
test:
	pytest tests/ -v --cov=src/bot --cov-report=html

test-fast:
	pytest tests/ -v -x

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ --line-length=100
	isort src/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

# Docker commands
docker-build:
	docker build -t bybit-trading-bot .

docker-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

docker-prod:
	docker-compose up -d --build

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f trading-bot

# Database management
db-init:
	python -c "from src.bot.database import DatabaseManager; from src.bot.config import Config; dm = DatabaseManager(Config.from_file('config/config.yaml').database); dm.initialize()"

db-migrate:
	alembic upgrade head

db-reset:
	python -c "from src.bot.database import DatabaseManager; from src.bot.config import Config; dm = DatabaseManager(Config.from_file('config/config.yaml').database); dm.reset_database()"

# Backtesting
backtest:
	python -m src.bot.main --backtest-only

# Production deployment
deploy:
	docker-compose -f docker-compose.yml up -d --build
	docker-compose logs -f

# Health check
health:
	curl -f http://localhost:8501/health || echo "Service not responding"

# View logs
logs:
	tail -f logs/trading_bot.log

# Setup development environment
setup-dev:
	python -m venv venv
	./venv/Scripts/activate && pip install -r requirements.txt
	cp .env.example .env
	@echo "Please edit .env file with your API credentials"