
VENV_NAME=venv

all: venv

venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: ./requirements.txt
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	$(VENV_NAME)/bin/pip install -r ./requirements.txt
	touch $(VENV_NAME)/bin/activate

install:
	$(VENV_NAME)/bin/pip install -e .

clean:
	rm -rf $(VENV_NAME)
	find . -type f -name "*.pyc" -delete

.PHONY: all venv install clean
