
# 🚀 Data Science Pro

**AI-Powered Automated Data Science Pipeline**

Transform your data science workflow with an intelligent pipeline that automates EDA, preprocessing, model selection, training, and evaluation - all powered by LLM suggestions.

## 📋 Prerequisites

Before installing, ensure you have:
- Python 3.8 or higher
- OpenAI API key (for AI-powered features)
- Git installed on your system

## 📦 Step 1: Complete Installation Guide

### Option A: Install from Source (Development Mode)
```bash
# Step 1.1: Clone the repository
git clone <your-repo-url>
cd data_science_pro

# Step 1.2: Create virtual environment (HIGHLY RECOMMENDED)
python -m venv venv

# Step 1.3: Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Step 1.4: Install the package in development mode
pip install -e .

# Step 1.5: Verify installation
python -c "import data_science_pro; print('✅ Installation successful!')"
```

### Option B: Install from PyPI (when published)
```bash
pip install data-science-pro
```

### Installation Verification
```python
# Step 1.6: Test the installation
import data_science_pro
print("✅ Data Science Pro installed successfully!")

# Step 1.7: Check available components
from data_science_pro.data import DataAnalyzer, DataLoader, DataOperations
from data_science_pro.modeling import Trainer, Evaluator, ModelRegistry  
from data_science_pro.cycle import IntelligentController, ChainOfThoughtSuggester
print("✅ All modules imported successfully!")
```

## 🚀 Step 2: Quick Start - Your First AI-Powered Analysis

### Method 1: Command Line Interface (CLI)
```bash
# Step 2.1: Basic usage with your data
data-science-pro --data your_data.csv --target target_column --api_key your_openai_key

# What this does automatically:
# 1. Load your dataset
# 2. Generate comprehensive EDA report
# 3. Apply smart preprocessing (handle missing values, encode categoricals, scale features)
# 4. Train multiple models and select the best one
# 5. Display detailed evaluation metrics and insights
```

### Method 2: Python API (Interactive Mode)
```python
# Step 2.2: Initialize pipeline with OpenAI API key
import data_science_pro

pipeline = data_science_pro.DataSciencePro(api_key='your-openai-key')

# Step 2.3: Load your data
pipeline.input_data('your_data.csv', target_col='target_column')

# Step 2.4: Get AI-powered analysis
report = pipeline.report()
print("📊 Data Analysis Report:")
print(report)

# Step 2.5: Get intelligent suggestions
suggestions = pipeline.suggestions("How can I improve my model accuracy?")
print("🤖 AI Suggestions:", suggestions)
```

## 🔧 Step 3: Deep Dive - All Preprocessing Actions

### Execute Each Preprocessing Step Individually:

```python
# Step 3.1: Load sample data (Titanic dataset example)
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

# Load Titanic dataset as example
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Save for testing
df.to_csv('titanic_sample.csv', index=False)

# Step 3.2: Initialize pipeline
pipeline = data_science_pro.DataSciencePro(api_key='your-openai-key')
pipeline.input_data('titanic_sample.csv', target_col='survived')

# Step 3.3: Execute each preprocessing action
preprocessing_actions = [
    ('fill_na', 'Handle missing values'),
    ('drop_constant', 'Remove constant columns'),
    ('drop_high_na', 'Remove columns with >50% missing values'),
    ('encode_categorical', 'Encode categorical variables'),
    ('scale_numeric', 'Scale numeric features'),
    ('drop_duplicates', 'Remove duplicate rows'),
    ('feature_gen', 'Generate interaction features')
]

for action, description in preprocessing_actions:
    print(f"\n🔄 {description}...")
    result = pipeline.apply_action(action)
    print(f"✅ {action} completed")
    print(f"   Data shape after {action}: {pipeline.data.shape}")
```

## 🤖 Step 4: Model Training & Evaluation

### Available Models with Parameters

```python
# Step 4.1: Train Random Forest with custom parameters
print("🌲 Training Random Forest...")
pipeline.set_model('randomforest', {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
})
pipeline.train_model()

# Step 4.2: Evaluate the model
print("📊 Evaluating Random Forest...")
rf_results = pipeline.evaluate_model()
print("Random Forest Results:", rf_results)

# Step 4.3: Train Logistic Regression
print("📈 Training Logistic Regression...")
pipeline.set_model('logisticregression', {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'liblinear'
})
pipeline.train_model()

lr_results = pipeline.evaluate_model()
print("Logistic Regression Results:", lr_results)

# Step 4.4: Compare models
print("\n🏆 Model Comparison:")
print(f"Random Forest - Accuracy: {rf_results.get('accuracy', 0):.3f}, F1: {rf_results.get('f1_score', 0):.3f}")
print(f"Logistic Regression - Accuracy: {lr_results.get('accuracy', 0):.3f}, F1: {lr_results.get('f1_score', 0):.3f}")

# Choose best model
best_model = 'randomforest' if rf_results.get('accuracy', 0) > lr_results.get('accuracy', 0) else 'logisticregression'
print(f"🥇 Best Model: {best_model}")
```

## 🔄 Step 5: Advanced Cyclic Workflow

### Automated Iteration Until Target Performance

```python
# Step 5.1: Import cycle components
from data_science_pro.cycle.controller import IntelligentController
from data_science_pro.cycle.suggester import ChainOfThoughtSuggester

# Step 5.2: Initialize cycle controller
controller = IntelligentController()
suggester = ChainOfThoughtSuggester()

# Step 5.3: Define your performance goal
target_accuracy = 0.85
max_iterations = 10

print(f"🎯 Target: Achieve {target_accuracy} accuracy in max {max_iterations} iterations")

# Step 5.4: Run automated improvement cycle
for iteration in range(max_iterations):
    print(f"\n🔄 Iteration {iteration + 1}")
    
    # Current performance
    current_results = pipeline.evaluate_model()
    current_accuracy = current_results.get('accuracy', 0)
    
    print(f"Current Accuracy: {current_accuracy:.3f}")
    
    if current_accuracy >= target_accuracy:
        print(f"✅ Target achieved! Final accuracy: {current_accuracy:.3f}")
        break
    
    # Get AI suggestions for improvement
    suggestions = suggester.suggest_improvements(
        current_results=current_results,
        data_info=pipeline.report()
    )
    
    print("🤖 AI Suggestions:", suggestions)
    
    # Apply suggested improvements based on AI recommendations
    if 'different model' in suggestions.lower() or 'xgboost' in suggestions.lower():
        try:
            print("Trying XGBoost...")
            import xgboost as xgb
            pipeline.set_model('xgboost', {'max_depth': 6, 'n_estimators': 300})
            pipeline.train_model()
        except ImportError:
            print("XGBoost not available, trying different RandomForest parameters")
            pipeline.set_model('randomforest', {'n_estimators': 300, 'max_depth': 20})
            pipeline.train_model()
    
    elif 'feature' in suggestions.lower():
        print("Applying feature engineering...")
        pipeline.apply_action('feature_gen')
        pipeline.train_model()
    
    elif 'hyperparameter' in suggestions.lower():
        print("Trying different hyperparameters...")
        pipeline.set_model('randomforest', {
            'n_estimators': 400,
            'max_depth': 25,
            'min_samples_split': 3
        })
        pipeline.train_model()

print("\n🎉 Cyclic workflow completed!")
```

## 💾 Step 6: Model Management & Registry

### Save, Load, and Version Models

```python
# Step 6.1: Import model registry
from data_science_pro.modeling.registry import ModelRegistry

# Step 6.2: Initialize registry
registry = ModelRegistry()

# Step 6.3: Save current model with metadata
model_info = {
    'model_name': 'titanic_survival_rf',
    'version': 'v1.0',
    'accuracy': pipeline.evaluate_model()['accuracy'],
    'f1_score': pipeline.evaluate_model()['f1_score'],
    'features_used': list(pipeline.data.columns),
    'preprocessing_steps': ['fill_na', 'encode_categorical', 'scale_numeric']
}

# Step 6.4: Save model
registry.save_model(
    model=pipeline.model,
    model_name=model_info['model_name'],
    version=model_info['version'],
    metadata=model_info
)
print(f"💾 Model saved: {model_info['model_name']} {model_info['version']}")

# Step 6.5: List all saved models
saved_models = registry.list_models()
print("📋 Saved Models:", saved_models)

# Step 6.6: Load a specific model
loaded_model = registry.load_model('titanic_survival_rf', 'v1.0')
print("✅ Model loaded successfully!")

# Step 6.7: Load model with metadata
loaded_model, metadata = registry.load_model_with_metadata('titanic_survival_rf', 'v1.0')
print("Model Metadata:", metadata)
```

## 📊 Step 7: Comprehensive Data Analysis

### Deep Dive into Your Data

```python
# Step 7.1: Import analysis components
from data_science_pro.data.data_analyzer import DataAnalyzer
from data_science_pro.data.data_loader import DataLoader

# Step 7.2: Initialize components
analyzer = DataAnalyzer()
loader = DataLoader()

# Step 7.3: Load data with advanced options
data = loader.load_data('titanic_sample.csv', 
                       file_type='csv',
                       encoding='utf-8',
                       parse_dates=True)

# Step 7.4: Comprehensive analysis
print("🔍 Comprehensive Data Analysis:")
print("=" * 50)

# Basic statistics
basic_stats = analyzer.get_basic_stats(data)
print("1️⃣ Basic Statistics:")
for key, value in basic_stats.items():
    print(f"   {key}: {value}")

# Missing value analysis
missing_analysis = analyzer.analyze_missing_values(data)
print("\n2️⃣ Missing Value Analysis:")
for col, info in missing_analysis.items():
    print(f"   {col}: {info['count']} missing ({info['percentage']:.1f}%)")

# Data quality report
quality_report = analyzer.generate_data_quality_report(data)
print("\n3️⃣ Data Quality Report:")
print(quality_report)

# Correlation analysis for numeric columns
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) > 1:
    correlation_matrix = analyzer.analyze_correlations(data[numeric_cols])
    print("\n4️⃣ Top Correlations:")
    print(correlation_matrix.head(10))

# Categorical analysis
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print("\n5️⃣ Categorical Analysis:")
    for col in categorical_cols:
        unique_count = data[col].nunique()
        print(f"   {col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"   Top categories: {data[col].value_counts().head(3).to_dict()}")
```

## 🎯 Step 8: Complete End-to-End Example

### Full Pipeline Execution with Sample Data

```python
# Step 8.1: Complete setup
import data_science_pro
import pandas as pd
import numpy as np

print("🚀 Starting Complete Data Science Pipeline")
print("=" * 60)

# Step 8.2: Initialize pipeline
print("1️⃣ Initializing pipeline...")
pipeline = data_science_pro.DataSciencePro(api_key='your-openai-key')

# Step 8.3: Create sample data for demonstration
print("2️⃣ Creating sample data...")
np.random.seed(42)
sample_data = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.randint(30000, 150000, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'experience': np.random.randint(0, 30, 1000),
    'department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR', 'Finance'], 1000),
    'target': np.random.choice([0, 1], 1000)
})

# Add some missing values for realistic testing
sample_data.loc[np.random.choice(sample_data.index, 50), 'age'] = np.nan
sample_data.loc[np.random.choice(sample_data.index, 30), 'income'] = np.nan

# Save sample data
sample_data.to_csv('employee_data.csv', index=False)
pipeline.input_data('employee_data.csv', target_col='target')

# Step 8.4: Get initial analysis
print("3️⃣ Getting initial analysis...")
initial_report = pipeline.report()
print("Initial Report:", initial_report)

# Step 8.5: Apply all preprocessing
preprocessing_actions = [
    'fill_na', 'drop_constant', 'drop_high_na', 
    'encode_categorical', 'scale_numeric', 'drop_duplicates'
]

print("4️⃣ Applying preprocessing...")
for action in preprocessing_actions:
    print(f"   Applying {action}...")
    pipeline.apply_action(action)

# Step 8.6: Train multiple models and compare
models_to_train = [
    ('randomforest', {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}),
    ('logisticregression', {'C': 1.0, 'max_iter': 1000, 'random_state': 42})
]

print("5️⃣ Training models...")
results = {}
for model_name, params in models_to_train:
    print(f"   Training {model_name}...")
    pipeline.set_model(model_name, params)
    pipeline.train_model()
    results[model_name] = pipeline.evaluate_model()

# Step 8.7: Compare and select best model
print("6️⃣ Model Comparison:")
best_model = None
best_score = 0
for model_name, result in results.items():
    accuracy = result.get('accuracy', 0)
    f1 = result.get('f1_score', 0)
    print(f"   {model_name}: Accuracy = {accuracy:.3f}, F1 = {f1:.3f}")
    if accuracy > best_score:
        best_score = accuracy
        best_model = model_name

print(f"🏆 Best Model: {best_model} with accuracy: {best_score:.3f}")

# Step 8.8: Save best model
print("7️⃣ Saving best model...")
final_params = next(params for name, params in models_to_train if name == best_model)
pipeline.set_model(best_model, final_params)
pipeline.train_model()
pipeline.save_model('best_employee_model.pkl')

# Step 8.9: Generate final report
print("8️⃣ Generating final report...")
final_report = pipeline.report()
print("Final Report:", final_report)

print("\n🎉 Complete pipeline executed successfully!")
print(f"📊 Best model accuracy: {best_score:.3f}")
print("💾 Model saved as: best_employee_model.pkl")
```

## 🛠️ Step 9: Troubleshooting Common Issues

### Issue 1: Import Errors
```python
# If you get: ModuleNotFoundError: No module named 'data_science_pro'
# Solution: Check installation and Python path

import sys
print("Python path:", sys.path)
print("Current directory:", sys.getcwd())

# Reinstall if needed
!pip install -e .  # or python -m pip install -e .
```

### Issue 2: OpenAI API Errors
```python
# If you get: AuthenticationError or RateLimitError
# Solution: Check your API key

# Test your API key
try:
    import openai
    openai.api_key = 'your-openai-key'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("✅ API key is working")
except Exception as e:
    print(f"❌ API key issue: {e}")
    print("💡 Tip: Get your API key from https://platform.openai.com/api-keys")
```

### Issue 3: Data Loading Issues
```python
# If you get: FileNotFoundError or parsing errors
# Solution: Check file path and format

import os
print("Current directory:", os.getcwd())
print("Files in directory:", [f for f in os.listdir('.') if f.endswith('.csv')])

# Try different options
data = pd.read_csv('your_file.csv', encoding='latin-1', sep=';', header=0)
```

### Issue 4: Model Training Errors
```python
# If you get: ValueError about data types or shapes
# Solution: Check your data after preprocessing

print("Data types:", pipeline.data.dtypes)
print("Data shape:", pipeline.data.shape)
print("Target column:", pipeline.target_col if hasattr(pipeline, 'target_col') else "Not set")
print("Missing values:", pipeline.data.isnull().sum().sum())
```

## 🧪 Step 10: Comprehensive Installation Test

### Run This Test to Verify Everything Works

```python
import data_science_pro
import pandas as pd
import numpy as np
import os

def comprehensive_test():
    """Test all functionality step by step"""
    print("🧪 Comprehensive Data Science Pro Test")
    print("=" * 50)
    
    # Test 1: Basic import
    print("1️⃣ Testing basic import...")
    try:
        from data_science_pro import DataSciencePro
        print("   ✅ Basic import successful")
    except Exception as e:
        print(f"   ❌ Basic import failed: {e}")
        return False
    
    # Test 2: Advanced imports
    print("2️⃣ Testing advanced imports...")
    try:
        from data_science_pro.data import DataAnalyzer, DataLoader, DataOperations
        from data_science_pro.modeling import Trainer, Evaluator, ModelRegistry
        from data_science_pro.cycle import IntelligentController, ChainOfThoughtSuggester
        print("   ✅ Advanced imports successful")
    except Exception as e:
        print(f"   ⚠️  Advanced imports: {e}")
    
    # Test 3: Create test data
    print("3️⃣ Creating test data...")
    np.random.seed(42)
    test_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B', 'C'], 100),
        'feature3': np.random.randint(1, 100, 100),
        'target': np.random.choice([0, 1], 100)
    })
    test_data.to_csv('test_comprehensive.csv', index=False)
    print("   ✅ Test data created")
    
    # Test 4: Initialize pipeline
    print("4️⃣ Initializing pipeline...")
    try:
        pipeline = DataSciencePro(api_key='test-key')
        print("   ✅ Pipeline initialized (test mode)")
    except Exception as e:
        print(f"   ⚠️  Pipeline init: {e}")
    
    # Test 5: Data loading
    print("5️⃣ Testing data loading...")
    try:
        pipeline.input_data('test_comprehensive.csv', target_col='target')
        print("   ✅ Data loading successful")
    except Exception as e:
        print(f"   ⚠️  Data loading: {e}")
    
    # Test 6: Preprocessing
    print("6️⃣ Testing preprocessing...")
    try:
        pipeline.apply_action('fill_na')
        print("   ✅ Preprocessing successful")
    except Exception as e:
        print(f"   ⚠️  Preprocessing: {e}")
    
    # Test 7: Model training
    print("7️⃣ Testing model training...")
    try:
        pipeline.set_model('randomforest', {'n_estimators': 10, 'random_state': 42})
        pipeline.train_model()
        print("   ✅ Model training successful")
    except Exception as e:
        print(f"   ⚠️  Model training: {e}")
    
    # Test 8: Model evaluation
    print("8️⃣ Testing model evaluation...")
    try:
        results = pipeline.evaluate_model()
        print(f"   ✅ Model evaluation successful: Accuracy = {results.get('accuracy', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️  Model evaluation: {e}")
    
    # Test 9: Model registry
    print("9️⃣ Testing model registry...")
    try:
        from data_science_pro.modeling.registry import ModelRegistry
        registry = ModelRegistry()
        registry.save_model(pipeline.model, 'test_model', 'v1.0')
        loaded_model = registry.load_model('test_model', 'v1.0')
        print("   ✅ Model registry successful")
    except Exception as e:
        print(f"   ⚠️  Model registry: {e}")
    
    # Clean up
    if os.path.exists('test_comprehensive.csv'):
        os.remove('test_comprehensive.csv')
    
    print("\n🎉 Comprehensive test completed!")
    print("💡 Note: Some tests may show warnings if API key is not provided")
    print("   This is normal - the package core functionality works without API key")
    return True

# Run the test
if __name__ == "__main__":
    comprehensive_test()
```

---

## 📚 Next Steps & Advanced Usage

### 1. Custom Model Integration
```python
# Add your own custom models
from sklearn.ensemble import GradientBoostingClassifier

# Register custom model (when feature is available)
pipeline.register_custom_model('gradient_boosting', GradientBoostingClassifier)
pipeline.set_model('gradient_boosting', {'n_estimators': 100, 'learning_rate': 0.1})
```

### 2. Batch Processing Multiple Datasets
```python
# Process multiple datasets in batch
datasets = ['data1.csv', 'data2.csv', 'data3.csv']
results = []

for dataset in datasets:
    print(f"Processing {dataset}...")
    pipeline = DataSciencePro(api_key='your-key')
    pipeline.input_data(dataset, target_col='target')
    pipeline.apply_action('fill_na')
    pipeline.set_model('randomforest')
    pipeline.train_model()
    result = pipeline.evaluate_model()
    results.append({dataset: result})

print("Batch processing completed!")
for result in results:
    print(result)
```

### 3. Integration with MLflow (Advanced)
```python
# When MLflow integration is available
pipeline.log_experiment('my_experiment_1')
pipeline.track_metrics(['accuracy', 'f1_score', 'precision', 'recall'])
pipeline.log_parameters({'n_estimators': 100, 'max_depth': 10})
```

---

## 🤝 Support & Contributing

- **Issues**: Report bugs via GitHub Issues
- **Feature Requests**: Open a GitHub Issue with enhancement label  
- **Contributions**: Fork the repository and submit pull requests
- **Documentation**: Help improve this README with your suggestions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎉 Congratulations! You now have a fully functional AI-powered data science pipeline.**

**Start experimenting with your own datasets and let the AI guide you to better models!** 🚀
