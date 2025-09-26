# data_science_pro

A modular, LLM-powered data science pipeline for automated EDA, preprocessing, model selection, training, and testing.

## Installation

```bash
pip install .
```

## Usage

```bash
data-science-pro --help
```

Or run the pipeline interactively:

```bash
python pipeline.py
```

## Features
- Automated EDA and preprocessing
- LLM-powered suggestions and cyclic workflow
- Model selection, training, evaluation, and saving
- CLI entrypoint for easy usage

## Requirements
See `requirements.txt` for dependencies.

## Project Structure
- `api/` - LLM connector
- `cycle/` - Suggestion and control logic
- `data/` - Data loading, analysis, operations
- `modeling/` - Model training, evaluation, registry
- `utils/` - Utility files
- `pipeline.py` - Main pipeline class
- `test.py` - Example/test script
