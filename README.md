# Experiment

### Build API-Integrated-LLM Package

Start with creating a new virtual environment.

```bash
conda create --name ail python=3.10 -y
conda activate ail
```

The tools in `agent-gym` can be conveniently utilized as Python packages. To build a local package, clone the repor and execute the following commands 
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

Conversational AI Gym Tool

options:
  -h, --help            show this help message and exit
  -m {default,evaluator,scorer}, --mode {default,evaluator,scorer}
                        Cli mode
  -rt ROOT, --root ROOT
                        Dataset root absolute path
  -ig, --ignore, --no-ignore
                        Ignore data points marked as "ignore"
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
        ├── examples_icl.json
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