# Variables
VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# Default target
all: setup run

# Create virtual environment
venv:
	python3 -m venv $(VENV)

# Install dependencies
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Setup everything
setup: install

# Run LLM simulation
run:
	$(PYTHON) ./llm_simulation/src/get_data.py

# Compare results
compare:
	$(PYTHON) llm_simulation/src/compare_data.py

# Clean environment
clean:
	rm -rf $(VENV)