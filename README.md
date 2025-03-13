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
usage: api-integrated-llm [-h] [-m {default,evaluator,scorer,parser,metrics_aggregator}] [-rt ROOT] [-eof EVALUATOR_OUTPUT_FOLDER]
                          [-pif PARSER_INPUT_FOLDER] [-of OUTPUT_FOLDER] [-sif SCORER_INPUT_FOLDER] [-dsf DATABASES_FOLDER]
                          [-maif METRICS_AGGREGATOR_INPUT_FOLDER] [-sf SOURCE_FOLDER] [-ig | --ignore | --no-ignore] [-asy | --use_async | --no-use_async]
                          [-si | --single_intent | --no-single_intent] [-er | --random_example | --no-random_example] [-nr NUMBER_RANDOM_EXAMPLE]
                          [-ep EXAMPLE_FILE_PATH] [-mcp LANGUAGE_MODEL_CONFIGURATION_FILE_PATH]

API Integrated LLM CLI

options:
  -h, --help            show this help message and exit
  -m {default,evaluator,scorer,parser,metrics_aggregator}, --mode {default,evaluator,scorer,parser,metrics_aggregator}
                        Cli mode
  -rt ROOT, --root ROOT
                        Dataset root folder absolute path
  -eof EVALUATOR_OUTPUT_FOLDER, --evaluator_output_folder EVALUATOR_OUTPUT_FOLDER
                        Evaluator output folder path
  -pif PARSER_INPUT_FOLDER, --parser_input_folder PARSER_INPUT_FOLDER
                        Parser input folder path
  -of OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Output folder path
  -sif SCORER_INPUT_FOLDER, --scorer_input_folder SCORER_INPUT_FOLDER
                        Scorer input folder path
  -dsf DATABASES_FOLDER, --databases_folder DATABASES_FOLDER
                        Databases folder path
  -maif METRICS_AGGREGATOR_INPUT_FOLDER, --metrics_aggregator_input_folder METRICS_AGGREGATOR_INPUT_FOLDER
                        Metrics aggregator input folder path
  -sf SOURCE_FOLDER, --source_folder SOURCE_FOLDER
                        Source folder path
  -ig, --ignore, --no-ignore
                        Ignore data points marked as "ignore"
  -asy, --use_async, --no-use_async
                        Use asynchronous operations
  -si, --single_intent, --no-single_intent
                        Single intent dataset
  -er, --random_example, --no-random_example
                        Create examples in prompts by sampling source data randomly
  -nr NUMBER_RANDOM_EXAMPLE, --number_random_example NUMBER_RANDOM_EXAMPLE
                        The number of examples sampled from source data randomly
  -ep EXAMPLE_FILE_PATH, --example_file_path EXAMPLE_FILE_PATH
                        The absolute path for an example file for a prompt
  -mcp LANGUAGE_MODEL_CONFIGURATION_FILE_PATH, --language_model_configuration_file_path LANGUAGE_MODEL_CONFIGURATION_FILE_PATH
                        The absolute path for a language model configuration
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

- `-m`: Tool mode
- `-pif`: Path to the folder containing parser input files (evaluator output files) in JSONL format.
- `-of`: Output folder path for the parsed results.
- `-si`: Indicates that the input files are for single intent detection.

### Input/Output:

- Both input and output files are in JSONL format, following the `EvaluationOutputResponseDataUnit` schema defined in `api_integrated_llm/data_models/source_models.py`.
- The parser sets `predicted_function_calls`, `gold_function_calls`, and `num_preciedtion_parsing_errors` fields in the output files.

### Example Usage:

```bash
api-integrated-llm -m parser -pif /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/tests/data/test_output/evaluation/llm -of /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/parsing -si
```

### Example Input/Output Files:

- Example input files for the parser are located at `tests/data/test_output/evaluation/llm`.
- Example output files for the parser are located at `tests/data/test_output/parsing`.


## Scorer

### Usage:

Run the scorer with the following command:

```bash
api-integrated-llm -m scorer -sif <INPUT_FOLDER_PATH> -of <OUTPUT_FOLDER_PATH> -si
```

### Flags: 

- `-m`: Tool mode
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

## Win Rate Calculator

### Pre-requisites

This win rate calculator, found at `api-integrated-llm-experiment`, is based on Ben Elder's code (https://github.ibm.com/AI4BA/invocable-api-hub/blob/sql/invocable_api_hub/driver/run_example.py). It has been significantly modified for use in `api-integrated-llm-experiment`.

To use the win rate calculator, you need three things:
1. Source data file (retrieved from LLMs or Agents)
2. Output files from evaluation or parsing
3. Database folder

Ensure that your source data contains the following:
- `QuerySourceDataModel` in `api_integrated_llm/data_models/source_models.py` should have a unique `sample_id`.
- `QuerySourceModel` in `api_integrated_llm/data_models/source_models.py` should have a valid dataset name.

The win rate calculation algorithm uses `sample_id` to locate necessary data in the source data, and the dataset's name is used to determine which folder contains the corresponding database contents.

### Usage:

To use the win rate calculator, run the following command:

```bash
api-integrated-llm -m scorer -sif <INPUT_FOLDER_PATH> -of <OUTPUT_FOLDER_PATH> -dsf <DATABASES_FOLDER_PATH> -sf <EVALUATION_SOURCE_FOLDER_PATH> --si
```

### Input/Output:

Ensure that `DATABASES_FOLDER_PATH` and `EVALUATION_SOURCE_FOLDER_PATH` are provided for the calculator to work.

- `EVALUATION_SOURCE_FOLDER_PATH`: Path to a folder containing source files for benchmarking, e.g., `tests/data/source/evaluation_win_rate`. The source files follow the `QuerySourceModel` defined in `api_integrated_llm/data_models/source_models.py`.
- `DATABASES_FOLDER_PATH`: Absolute folder path containing databases, e.g., a file structure with folders representing databases inside `DATABASES_FOLDER`. For example,

```
DATABASES_FOLDER
├── california_schools
│   ├── california_schools.sqlite
│   └── database_description
│       ├── frpm.csv
│       ├── satscores.csv
│       └── schools.csv
├── card_games
│   ├── card_games.sqlite
│   └── database_description
│       ├── cards.csv
│       ├── foreign_data.csv
│       ├── legalities.csv
│       ├── rulings.csv
│       ├── set_translations.csv
│       └── sets.csv
├── codebase_community
```

- Output files are in JSON format, following the `ScorerOuputModel` schema defined in `api_integrated_llm/data_models/scorer_models.py`.

### Example Usage:

```bash
api-integrated-llm -m scorer -sif /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/parsing -of /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/scoring -dsf /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/tests/data/source/databases -sf /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/tests/data/source/evaluation_win_rate -si
```

## Metrics Aggregator

### Usage:

Run the metrics aggregator with the following command:

```bash
api-integrated-llm -m metrics_aggregator -maif <METRICS_AGGREGATOR_INPUT_FOLDER> -of <METRICS_AGGREGATOR_OUTPUT_FOLDER>
```

### Flags: 

- `-m`: Tool mode
- `-maif`: Path to the folder containing metrics aggregator input files, which are scorer output files JSON format.
- `-of`: Output folder path for the metrics aggregator results.

### Input/Output:

- input files are in JSON format, following the `ScorerOuputModel` schema defined in `api_integrated_llm/data_models/scorer_models.py`. See an example input files at `tests/data/test_output/scoring`
- output files are in JSON format, following the `AggegatorOutputModel` schema defined in `api_integrated_llm/data_models/scorer_models.py`. See an example output file at `tests/data/test_output/metrics_aggregation/metrics_aggregation_03_06_2025_15_23_27.json`.

### Example Usage:

```bash
api-integrated-llm -m metrics_aggregator -maif /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/tests/data/test_output/scoring -of /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/
```