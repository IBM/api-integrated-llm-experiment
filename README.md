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
usage: api-integrated-llm [-h] [-m {default,evaluator,scorer,parser}] [-rt ROOT]
                          [-eof EVALUATOR_OUTPUT_FOLDER] [-pif PARSER_INPUT_FOLDER] [-of OUTPUT_FOLDER]
                          [-sif SCORER_INPUT_FOLDER] [-ig | --ignore | --no-ignore]
                          [-si | --single_intent | --no-single_intent]
                          [-er | --random_example | --no-random_example] [-nr NUMBER_RANDOM_EXAMPLE]
                          [-ep EXAMPLE_FILE_PATH]

API Integrated LLM CLI

options:
  -h, --help            show this help message and exit
  -m {default,evaluator,scorer,parser}, --mode {default,evaluator,scorer,parser}
                        Cli mode
  -rt ROOT, --root ROOT
                        Dataset root absolute path
  -eof EVALUATOR_OUTPUT_FOLDER, --evaluator_output_folder EVALUATOR_OUTPUT_FOLDER
                        Evaluator output folder path
  -pif PARSER_INPUT_FOLDER, --parser_input_folder PARSER_INPUT_FOLDER
                        Parser input folder path
  -of OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Output folder path
  -sif SCORER_INPUT_FOLDER, --scorer_input_folder SCORER_INPUT_FOLDER
                        Scorer input folder path
  -ig, --ignore, --no-ignore
                        Ignore data points marked as "ignore"
  -si, --single_intent, --no-single_intent
                        Single intent dataset
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


## Parser

### Usage:

Run the parser with the following command:

```bash
api-integrated-llm -m parser -pif <INPUT_FOLDER_PATH> -of <OUTPUT_FOLDER_PATH> -si
```

### Flags: 

- `-pif`: Path to the folder containing parser input files (evaluator output files) in JSONL format.
- `-of`: Output folder path for the parsed results.
- `-si`: Indicates that the input files are for single intent detection.

### Input/Output:

- Both input and output files are in JSONL format, following the `EvaluationOutputResponseDataUnit` schema defined in `api_integrated_llm/data_models/source_models.py`.
- The parser sets `predicted_function_calls`, `gold_function_calls`, and `num_preciedtion_parsing_errors` fields in the output files.

### Example Usage:

```bash
api-integrated-llm -m parser -pif /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/tests/data/test_output/evaluation/llm -of /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/parsing -si\n
```

### Example Input/Output Files:

- Example input files for the parser are located at `tests/data/test_output/evaluation/llm`.
- Example output files for the parser are located at `tests/data/test_output/parsing`.


## Scorer

Parser can be used by providing 1. the path of the folder (`-pif` flag) containing scorer input files (evaluator output files or parser output files) in JSONL and the output folder path (`of` flag). `-si` flag indicate that the input files are for single intent detection. Both input and output files are JSONL, and each line in the files follow `EvaluationOutputResponseDataUnit` schema defined in `api_integrated_llm/data_models/source_models.py`. `Parser` sets `predicted_function_calls`, `gold_function_calls`, and `num_preciedtion_parsing_errors` fields in `EvaluationOutputResponseDataUnit`. The following command shows how to use `Parser`.

### Usage:

Run the scorer with the following command:

```bash
api-integrated-llm -m parser -sif <INPUT_FOLDER_PATH> -of <OUTPUT_FOLDER_PATH> -si
```

### Flags: 

- `-sif`: Path to the folder containing scorer input files, which are 1. evaluator output files or 2. parser output files in JSONL format.
- `-of`: Output folder path for the scorer results.
- `-si`: Indicates that the input files are for single intent detection.

### Input/Output:

- Input files are in JSONL format, following the `EvaluationOutputResponseDataUnit` schema defined in `api_integrated_llm/data_models/source_models.py`.
- Keep in mind that Parser output files contain valid (non-null) `num_preciedtion_parsing_errors` values. Meanwhile, Evaluator output files contain null `num_preciedtion_parsing_errors` values.
- Output files are in JSON format, following the `ScorerOuputModel` schema defined in `api_integrated_llm/data_models/scorer_models.py`.

### Example Usage:

```bash
api-integrated-llm -m scorer -sif /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/parsing -of /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/scoring -si
```

### Example Input/Output Files:

- Example input files for the scorer are located at `tests/data/test_output/evaluation/llm` and `tests/data/test_output/parsing`.
- Example output files for the parser are located at `tests/data/test_output/scoring`.