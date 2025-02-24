# Experiment

### Build API-Integrated-LLM Package

Start with creating a new virtual environment.

```bash
conda create --name ail python=3.10 -y
conda activate ail
```

The tools in `api-integrated-llm` can be conveniently utilized as Python packages. To build a local package, clone the repor and execute the following commands 
from the project's root directory. 

```bash
git clone git@github.ibm.com:Jungkoo-Kang/api_integrated_llm_experiment.git
cd api_integrated_llm_experiment
```

Now, to build, run:

```bash
pip install -e .
python -m pip install --upgrade build
python -m build
```

## Installation Guide

To install the CLI, follow these steps:

1. Navigate to your Python virtual environment for API-Integrated-LLM CLI.
2. Run the following command to install the package from the built WHL file:

```bash
pip install <PATH_TO_BUILT_WHL_FILE>
```

3. To verify that the installation was successful, execute the following command to display the help menu:
```bash
api-integrated-llm -h
```

## Command Line Interface (CLI)

```
usage: api-integrated-llm [-h] [-m {default,evaluator,scorer}] [-rt ROOT] [-ig | --ignore | --no-ignore]
                          [-er | --random_example | --no-random_example] [-nr NUMBER_RANDOM_EXAMPLE] [-ep EXAMPLE_FILE_PATH]

API Integrated LLM CLI

options:
  -h, --help            show this help message and exit
  -m {default,evaluator,scorer}, --mode {default,evaluator,scorer}
                        Cli mode
  -rt ROOT, --root ROOT
                        Dataset root absolute path
  -ig, --ignore, --no-ignore
                        Ignore data points marked as "ignore"
  -er, --random_example, --no-random_example
                        Create examples in prompts by sampling source data randomly
  -nr NUMBER_RANDOM_EXAMPLE, --number_random_example NUMBER_RANDOM_EXAMPLE
                        The number of examples sampled from source data randomly
  -ep EXAMPLE_FILE_PATH, --example_file_path EXAMPLE_FILE_PATH
                        The absolute path for an example file for a prompt
```

The source folder should have the following structure:

```txt
.
└── source
    ├── configurations
    │   └── llm_configurations.json
    ├── evaluation
    │   └── <YOUR_DATA_HERE>
    └── prompts
        ├── examples
        │   └── examples.json
        └── prompts.json
```

To evaluate and score data, use the following command:

```bash
api-integrated-llm -rt <PATH_TO_DATA_SOURCE_FOLDER>
```

### Development

Install development dependencies.

```bash
pip install -e ".[test]"
pre-commit install
```