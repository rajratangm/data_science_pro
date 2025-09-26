
# 🚀 Data Science Pro

**AI-Powered Automated Data Science Pipeline**

Transform your data science workflow with an intelligent pipeline that automates EDA, preprocessing, model selection, training, and evaluation - all powered by LLM suggestions.

## ✨ Key Features

- **🤖 AI-Powered Suggestions**: Get intelligent recommendations for preprocessing, feature engineering, and model selection
- **📊 Automated EDA**: Generate comprehensive data analysis reports instantly
- **🔧 Smart Preprocessing**: Handle missing values, encode categoricals, scale features automatically
- **🎯 Model Selection**: LLM suggests optimal models and hyperparameters based on your data
- **📈 Training & Evaluation**: Train models with built-in evaluation metrics
- **💾 Model Registry**: Save and version your trained models
- **🔄 Cyclic Workflow**: Iterate until you achieve your desired performance metrics

## 📦 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd data_science_pro

# Install the package
pip install .
```

## 🚀 Quick Start

### Option 1: Command Line Interface (CLI)
```bash
# Basic usage
data-science-pro --data your_data.csv --target target_column --api_key your_openai_key

# This will:
# - Load your dataset
# - Generate EDA report
# - Apply preprocessing (drop NA, encode, scale)
# - Train a RandomForest model
# - Display evaluation metrics
```

### Option 2: Python API (Interactive)
```python
import data_science_pro

# Initialize pipeline with OpenAI API key
pipeline = data_science_pro.DataSciencePro(api_key='your-openai-key')

# Load your data
pipeline.input_data('your_data.csv', target_col='target_column')

# Get AI-powered analysis
report = pipeline.report()
print("📊 Data Analysis Report:")
print(report)

# Get intelligent suggestions
suggestions = pipeline.suggestions("How can I improve my model accuracy?")
print("🤖 AI Suggestions:", suggestions)
```

## 📋 Available Preprocessing Actions

| Action | Description |
|--------|-------------|
| `drop_na` | Remove rows with missing values |
| `fill_na` | Fill missing values (median for numeric, mode for categorical) |
| `encode_categorical` | One-hot encode categorical variables |
| `scale_numeric` | Standard scale numeric features |
| `drop_duplicates` | Remove duplicate rows |
| `drop_constant` | Remove columns with constant values |
| `drop_high_na` | Remove columns with >50% missing values |
| `feature_gen` | Generate interaction features |

## 🤖 Available Models

| Model | Parameters |
|-------|------------|
| `randomforest` | `{'n_estimators': 100, 'max_depth': 10}` |
| `logisticregression` | `{'C': 1.0, 'max_iter': 1000}` |

## 🎯 Complete Workflow Example

```python
import data_science_pro

# Initialize pipeline
pipeline = data_science_pro.DataSciencePro(api_key='your-openai-key')

# 1. Load data
pipeline.input_data('titanic.csv', target_col='Survived')

# 2. Get initial analysis
print("Initial Analysis:", pipeline.report())

# 3. Apply preprocessing
preprocessing_steps = ['drop_na', 'encode_categorical', 'scale_numeric']
for step in preprocessing_steps:
    print(f"Applying {step}...")
    pipeline.apply_action(step)

# 4. Train model with AI-suggested hyperparameters
pipeline.set_model('randomforest', {'n_estimators': 200, 'max_depth': 15})
pipeline.train()

# 5. Evaluate model
results = pipeline.evaluate()
print("📈 Model Performance:", results)

# 6. Save model
pipeline.save_model('titanic_model.pkl')
print("💾 Model saved successfully!")
```

## 🔄 Advanced: Cyclic Workflow

```python
from data_science_pro.cycle.controller import Controller

# Run automated cyclic workflow until target metric is achieved
controller = Controller()
pipeline.run_full_cycle(controller, metric_goal=0.85)
```

## 🛠️ Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'data_science_pro'`
**Solution**: Make sure you installed the package with `pip install .` and you're running from a different directory than the package source.

### Error: `NameError: name 'OneHotEncoder' is not defined`
**Solution**: This was a bug that's been fixed. Update your package installation.

### Error: `TypeError: input_data() got an unexpected keyword argument 'target'`
**Solution**: Use `target_col` instead of `target`:
```python
# ❌ Wrong
pipeline.input_data('data.csv', target='column_name')

# ✅ Correct
pipeline.input_data('data.csv', target_col='column_name')
```

### OpenAI API Issues
- Make sure your API key is valid and has credits
- Check your internet connection
- Verify the API key format: `sk-...`

## 📊 Example Output

```
📊 Data Analysis Report:
Dataset shape: (891, 12)
Target variable: Survived
Missing values: Age (177), Cabin (687), Embarked (2)
Data types: 5 numeric, 7 categorical

🤖 AI Suggestions:
"Based on your data, I recommend:
1. Fill missing Age values with median
2. Drop Cabin column due to high missingness (77%)
3. Encode Sex and Embarked as categorical
4. Consider RandomForest with n_estimators=200"

📈 Model Performance:
{'accuracy': 0.83, 'precision': 0.81, 'recall': 0.79, 'f1_score': 0.80}
```

## 🧪 Testing Your Installation

Run this quick test to verify everything works:

```python
import data_science_pro

# Test basic functionality
pipeline = data_science_pro.DataSciencePro(api_key='test-key')
print("✅ Package imported successfully!")

# Test with sample data
import pandas as pd
sample_data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': ['A', 'B', 'A', 'B', 'A'],
    'target': [0, 1, 0, 1, 0]
})

sample_data.to_csv('test_data.csv', index=False)
pipeline.input_data('test_data.csv', target_col='target')
print("✅ Data loading works!")

# Clean up
import os
os.remove('test_data.csv')
print("✅ All tests passed!")
```

## 🔧 Extending the Package

### Adding Custom Preprocessing
Edit `data_science_pro/data/data_operations.py`:

```python
def your_custom_operation(self, df, **kwargs):
    # Your preprocessing logic here
    return df
```

### Adding New Models
Edit the `set_model` method in `pipeline.py`:

```python
elif model_name.lower() == 'your_model':
    from sklearn.your_model import YourModel
    self.model_instance = YourModel(**hyperparams)
```

## 📚 Requirements

- Python 3.8+
- pandas
- scikit-learn
- langchain
- openai
- imbalanced-learn

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with LangChain for LLM integration
- Powered by scikit-learn for machine learning
- Inspired by automated ML pipelines

---

**Happy Data Science! 🎉** Start building smarter models with AI-powered assistance today!
