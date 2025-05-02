# Makefile for running and managing tuned2.ipynb

NOTEBOOK = tuned2.ipynb
SCRIPT = tuned2.py
OUTPUT = tuned2_output.ipynb

all: run

convert:
	jupyter nbconvert --to script $(NOTEBOOK) --output $(SCRIPT)

run:
	jupyter nbconvert --to notebook --execute $(NOTEBOOK) --output $(OUTPUT)

clean:
	rm -f $(SCRIPT) $(OUTPUT)

env:
	pip install -r requirements.txt

help:
	@echo "Usage:"
	@echo "  make convert   - Convert notebook to Python script"
	@echo "  make run       - Execute notebook and save output"
	@echo "  make clean     - Remove generated files"
	@echo "  make env       - Install dependencies from requirements.txt"
