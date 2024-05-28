VENV := .venv

SOURCE := nfs_ab_tool
TESTS := tests

start:
	. $(VENV)/bin/activate

setup:
	echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	$(MAKE) start
	pip install --upgrade pip
	pip install -r requirements.txt

delete_cash:
	@find . -type f -name '.DS_Store' -exec rm {} \;
	@find . -type d -name '__pycache__' -exec rm -rf {} \;
	@find . -type d -name '.pytest_cache' -exec rm -rf {} \;

clean:
	$(MAKE) delete_cash
	rm -rf $(VENV)

lint:
	echo "Running pylint check ..."
	pylint $(SOURCE) $(TESTS)


format:
	echo "Run formatting ..."
	isort $(SOURCE) $(TESTS)
	black $(SOURCE) $(TESTS)


test:
	echo "Running pytest checks...\t"
	PYTHONPATH=. pytest $(TESTS)

