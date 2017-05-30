.PHONY: clean data lint requirements train blobs train_blobs

PYTHON_INTERPRETER = python3

## Project specific
# DATA_FILE = "data/connectivity_10000_8x8.h5"
DATA_FILE = "data/test.h5"
BLOB_DATA_FILE = "data/blobs_10000_8x8.h5"
DATA_DIR = "data/"

WIDTH = 8
HEIGHT = 8
N_BLOBS = 4
N_EXAMPLES = 5000

## Install Python Dependencies
requirements: test_environment
	pip install -r requirements.txt

## Train model
train:
	$(PYTHON_INTERPRETER) src/models/conv_ca.py --num_layers=30 --state_size=14 --run=101 \
	--debug=True --width=14 --height=14 --test_fraction=0.2 --data_dir $(DATA_FILE) \
	--dense_logs=True --result_dir="../debug" --batch_size=24 --learning_rate=1e-4

viz:
	# $(PYTHON_INTERPRETER) main.py vizualization "../debug/14x14/lr=1e-04,layers=10,state=14/run101" 4
	# $(PYTHON_INTERPRETER) main.py vizualization "data/blobs_10000_8x8.h5" 3
	$(PYTHON_INTERPRETER) main.py vizualization ../debug/14x14/ 1

## Generate pbs job script
## pbs: WIP

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py -h 28 -w 28 -n 10000 -p 0.5 'random_walker' $(DATA_FILE)

boxplot_sm:
	$(PYTHON_INTERPRETER) main.py visualization ../results/8x8/lr=1e-03,layers=1,state=1/run101 1

blobs:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py -h $(HEIGHT) -w $(WIDTH) -n $(N_EXAMPLES) \
	-k $(N_BLOBS) -m 1 'blobs' $(DATA_DIR)


train_blobs:
	$(PYTHON_INTERPRETER) src/models/conv_ca.py --num_layers=10 --state_size=10 --run=101 \
	--debug=True --height=$(HEIGHT) --width=$(WIDTH) --test_fraction=0.2 --data_dir=$(BLOB_DATA_FILE) \
	--dense_logs=True --result_dir="../blob_results" --num_classes=$(N_BLOBS) --batch_size=500

test:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

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
