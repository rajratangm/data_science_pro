
# data_science_pro

A modular, LLM-powered data science pipeline for automated EDA, preprocessing, model selection, training, and testing.

## Overview
`data_science_pro` is designed to behave like an AI-powered junior data scientist. It automates the full data science workflow, including:
- Exploratory Data Analysis (EDA)
- Data preprocessing (handling missing values, encoding, scaling, feature engineering)
- Model selection and hyperparameter suggestion (using LLMs)
- Model training and evaluation
- Saving and loading models
- Interactive cyclic workflow with user input and LLM suggestions at every step

## How It Works
1. **Data Loading:** Load your dataset and specify the target column.
2. **EDA & Reporting:** The pipeline generates dynamic analysis reports using built-in analyzers.
3. **Preprocessing:** Automated or user-guided preprocessing using modular actions (drop NA, encode, scale, etc.), powered by LLM suggestions.
4. **Model Selection:** The LLM agent suggests suitable models and hyperparameters based on data analysis and user goals.
5. **Training:** The selected model is trained on the processed data.
6. **Evaluation:** The model is evaluated using standard metrics (accuracy, precision, recall, F1, etc.).
7. **Saving:** Trained models can be saved and versioned for future use.
8. **Cyclic Workflow:** The pipeline can repeat steps, allowing iterative improvement based on metrics and user feedback.

## Features
- Modular OOP design for easy extension
- LangChain-powered LLM agent for suggestions and decision-making
- CLI entrypoint for easy usage
- Handles both automated and interactive workflows
- Supports custom preprocessing, feature engineering, and model registry

## Installation
```bash
pip install .
```


## Usage

### CLI Example
Run the pipeline from the command line:
```bash
data-science-pro --data your_data.csv --target target_column --api_key your_openai_key
```
This will:
- Load your dataset
- Run EDA and print a report
- Apply basic preprocessing (drop NA, encode categoricals, scale numerics)
- Train a RandomForest model
- Print evaluation metrics

### Python API Example
Use the pipeline interactively in Python:
```python
from data_science_pro.pipeline import DataSciencePro

# Initialize pipeline with your OpenAI API key
pipeline = DataSciencePro(api_key='your-openai-key')

# Load data
pipeline.input_data('your_data.csv', 'target_column')

# Get EDA report
report = pipeline.report()
print(report)

# Get LLM-powered suggestions for next action
suggestion = pipeline.suggestions(user_query="How should I preprocess this data?", metrics=None)
print("LLM Suggestion:", suggestion)

# Apply preprocessing actions
pipeline.apply_action('drop_na')
pipeline.apply_action('encode_categorical')
pipeline.apply_action('scale_numeric')

# Model selection and training
pipeline.set_model('randomforest', {'n_estimators': 100})
pipeline.train()

# Evaluation
metrics = pipeline.evaluate()
print("Evaluation Metrics:", metrics)

# Save model
pipeline.save_model('model.joblib')
```

## What Can Be Done With This Project?

- **Automated EDA:** Instantly generate data analysis reports.
- **Preprocessing:** Handle missing values, encode categoricals, scale features, drop irrelevant columns, and engineer new features.
- **LLM-powered Suggestions:** Get dynamic, context-aware recommendations for preprocessing, feature engineering, and model selection.
- **Model Selection:** Use LLM to suggest optimal models and hyperparameters.
- **Training & Evaluation:** Train models and evaluate with standard metrics (accuracy, precision, recall, F1, etc.).
- **Model Registry:** Save and version trained models for future use.
- **Cyclic Workflow:** Iterate through EDA, preprocessing, training, and evaluation until desired metrics are achieved.
- **Interactive & Automated:** Use interactively (Python API) or automate via CLI.
- **Extensible:** Easily add new preprocessing steps, models, or evaluation metrics.

## Clear Guidance

1. **Install dependencies:**
	```bash
	pip install .
	```
2. **Prepare your data:**
	- CSV format recommended
	- Ensure target column is present
3. **Get your OpenAI API key:**
	- Required for LLM-powered suggestions
4. **Run the pipeline:**
	- Use CLI or Python API as shown above
5. **Iterate:**
	- Use LLM suggestions to improve preprocessing, feature engineering, and model selection
6. **Save and reuse models:**
	- Use the registry to save trained models

## Advanced Features
- Add custom preprocessing actions in `data_operations.py`
- Extend LLM agent prompts in `cycle/suggester.py`
- Integrate with other ML libraries or cloud services


## Project Structure
- `api/` - LLM connector
- `cycle/` - Suggestion and control logic
- `data/` - Data loading, analysis, operations
- `modeling/` - Model training, evaluation, registry
- `utils/` - Utility files
- `pipeline.py` - Main pipeline class
- `test.py` - Example/test script

## Requirements
See `requirements.txt` for dependencies.

## Contributing
Pull requests and issues are welcome! Please see the guidelines in the repository.

## License
MIT
