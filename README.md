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
usage: api-integrated-llm [-h] [-m {default,evaluator,scorer,parser,metrics_aggregator}] [-rt ROOT] [-eof EVALUATOR_OUTPUT_FOLDER] [-pif PARSER_INPUT_FOLDER] [-of OUTPUT_FOLDER]
                          [-sif SCORER_INPUT_FOLDER] [-maif METRICS_AGGREGATOR_INPUT_FOLDER] [-ig | --ignore | --no-ignore] [-igf IGNORE_FILE_PATH]
                          [-asy | --use_async | --no-use_async] [-si | --single_intent | --no-single_intent] [-atp | --add_tool_definition_to_prompt | --no-add_tool_definition_to_prompt]
                          [-er | --random_example | --no-random_example] [-nr NUMBER_RANDOM_EXAMPLE] [-ep EXAMPLE_FILE_PATH] [-mcp LANGUAGE_MODEL_CONFIGURATION_FILE_PATH]

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
  -maif METRICS_AGGREGATOR_INPUT_FOLDER, --metrics_aggregator_input_folder METRICS_AGGREGATOR_INPUT_FOLDER
                        Metrics aggregator input folder path
  -ig, --ignore, --no-ignore
                        Ignore data points marked as "ignore"
  -igf IGNORE_FILE_PATH, --ignore_file_path IGNORE_FILE_PATH
                        Ignore file path
  -asy, --use_async, --no-use_async
                        Use asynchronous operations
  -si, --single_intent, --no-single_intent
                        Single intent dataset
  -atp, --add_tool_definition_to_prompt, --no-add_tool_definition_to_prompt
                        Add tool definitions to prompt
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

Make sure to set environmental variables with api keys.

```bash
export RITS_API_KEY="<API_KEY>" # pragma: allowlist secret
export AZURE_OPENAI_API_KEY="<API_KEY>" # pragma: allowlist secret
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
api-integrated-llm -m parser -pif <INPUT_FOLDER_PATH> -of <OUTPUT_FOLDER_PATH>
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
api-integrated-llm -m parser -pif /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/tests/data/test_output/evaluation/llm -of /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/parsing
```

### Example Input/Output Files:

- Example input files for the parser are located at `tests/data/test_output/evaluation/llm`.
- Example output files for the parser are located at `tests/data/test_output/parsing`.


## Scorer

### Usage:

Run the scorer with the following command:

```bash
api-integrated-llm -m scorer -sif <INPUT_FOLDER_PATH> -of <OUTPUT_FOLDER_PATH>
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
api-integrated-llm -m scorer -sif /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/parsing -of /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/scoring
```

### Example Input/Output Files:

- Example input files for the scorer are located at `tests/data/test_output/evaluation/llm` and `tests/data/test_output/parsing`.
- Example output files for the parser are located at `tests/data/test_output/scoring`.

### Exclude Samples:

To exclude specific samples from processing, provide the absolute file path for a JSON file containing file names and sample IDs to ignore to the Scorer using the `-igf` flag. An example JSON file is available at `tests/data/source/auxiliary/ignore.json`.

```bash
api-integrated-llm -m scorer -sif <INPUT_FOLDER_PATH> -of <OUTPUT_FOLDER_PATH> -igf <IGORE_FILE_PATH>
```

Example command:

```bash
api-integrated-llm -m scorer -sif /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/parsing -of /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/output/scoring -igf /Users/jungkookang/Documents/projects/api_integrated_llm_experiment/tests/data/source/auxiliary/ignore.json
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