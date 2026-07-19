# Standard task interface across the portfolio: up / down / demo / test / logs.
.DEFAULT_GOAL := help
.PHONY: help up down logs ps benchmark demo test

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

up: ## Boot the full stack (server + 3 bank nodes)
	docker compose up --build

down: ## Stop the stack and remove volumes
	docker compose down -v

logs: ## Tail service logs
	docker compose logs -f

ps: ## Show running services
	docker compose ps

benchmark: ## Collaboration lift — federated AUC vs. each bank's solo model
	docker compose run --rm --no-deps server python -m benchmark

demo: ## Headline demo — membership-inference attack succeeds without DP, fails with DP
	docker compose run --rm --no-deps server python -m demo

test: ## Run the Python test suite
	pytest -q
