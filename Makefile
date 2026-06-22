.PHONY: test test-unit test-integration test-cov

test:
	python3 -m pytest

test-unit:
	python3 -m pytest -m unit

test-integration:
	python3 -m pytest -m integration


test-cov:
	python3 -m pytest --cov=. --cov-report=term-missing
