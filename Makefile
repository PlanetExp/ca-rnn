.PHONY: clean data lint requirements train

PYTHON_INTERPRETER = python3

## Project specific
DATA_FILE = "data/connectivity_10000_28x28.h5"

## Install Python Dependencies
requirements: test_environment
	pip install -r requirements.txt

## Train model
train:
	$(PYTHON_INTERPRETER) src/models/conv_ca.py --num_layers=10 --state_size=10 --run=101 \
	--debug=True --width=28 --height=28 --test_fraction=0.2 --data_dir $(DATA_FILE) \
	--dense_logs=True

## Generate pbs job script
## pbs: WIP

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py -h 28 -w 28 -n 10000 -p 0.5 $(DATA_FILE)

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Set up python interpreter environment
## create_environment:
	## $(PYTHON_INTERPRETER) -m venv venv
	## @pip install -q virtualenv virtualenvwrapper
	## @echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	## @bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	## @echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py
