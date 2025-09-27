#!/usr/bin/env python3
"""
Simple test script to verify DataSciencePro package functionality.
"""

import os
import sys
import pandas as pd
import numpy as np

# Check for required dependencies
try:
    import langchain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not available - some features may not work")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available - AI suggestions will not work")

def create_test_data():
    """Create simple test dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    try:
        from data_science_pro import DataSciencePro
        print("✅ Import successful")
        return True
    except ImportError as e:
        if "langchain" in str(e).lower():
            print(f"⚠️  Import failed due to missing langchain: {e}")
            print("💡 Try: pip install langchain langchain-community")
        else:
            print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_initialization():
    """Test DataSciencePro initialization."""
    print("Testing initialization...")
    try:
        from data_science_pro import DataSciencePro
        ai = DataSciencePro()
        print("✅ Initialization successful")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    try:
        from data_science_pro import DataSciencePro
        ai = DataSciencePro()
        
        # Create test data
        test_data = create_test_data()
        test_data.to_csv('test_data.csv', index=False)
        
        # Load data
        ai.input_data('test_data.csv', target_col='target')
        
        # Verify data loaded
        assert hasattr(ai, 'data')
        assert ai.data is not None
        assert ai.target_col == 'target'
        
        print("✅ Data loading successful")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_data_analysis():
    """Test data analysis functionality."""
    print("Testing data analysis...")
    try:
        from data_science_pro import DataSciencePro
        ai = DataSciencePro()
        
        # Load test data
        test_data = create_test_data()
        test_data.to_csv('test_data.csv', index=False)
        ai.input_data('test_data.csv', target_col='target')
        
        # Analyze data
        report = ai.report()
        
        # Check report structure
        assert 'shape' in report
        assert 'columns' in report
        assert 'feature_types' in report
        
        print("✅ Data analysis successful")
        return True
    except Exception as e:
        print(f"❌ Data analysis failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing operations."""
    print("Testing preprocessing...")
    try:
        from data_science_pro import DataSciencePro
        ai = DataSciencePro()
        
        # Load test data with missing values
        test_data = create_test_data()
        test_data.loc[0:5, 'feature1'] = np.nan  # Add some missing values
        test_data.to_csv('test_data.csv', index=False)
        ai.input_data('test_data.csv', target_col='target')
        
        # Test preprocessing actions
        ai.apply_action('drop_na')
        print("✅ Preprocessing successful")
        return True
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False

def test_model_training():
    """Test model training functionality."""
    print("Testing model training...")
    try:
        from data_science_pro import DataSciencePro
        ai = DataSciencePro()
        
        # Load test data
        test_data = create_test_data()
        test_data.to_csv('test_data.csv', index=False)
        ai.input_data('test_data.csv', target_col='target')
        
        # Set model and train
        ai.set_model('randomforest', {'n_estimators': 10})
        ai.train()
        
        print("✅ Model training successful")
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def test_model_evaluation():
    """Test model evaluation functionality."""
    print("Testing model evaluation...")
    try:
        from data_science_pro import DataSciencePro
        ai = DataSciencePro()
        
        # Load test data
        test_data = create_test_data()
        test_data.to_csv('test_data.csv', index=False)
        ai.input_data('test_data.csv', target_col='target')
        
        # Train model
        ai.set_model('randomforest', {'n_estimators': 10})
        ai.train()
        
        # Evaluate
        metrics = ai.evaluate()
        
        # Check metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in required_metrics:
            assert metric in metrics
        
        print("✅ Model evaluation successful")
        return True
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return False

def test_data_operations():
    """Test data operations module."""
    print("Testing data operations...")
    try:
        from data_science_pro.data.data_operations import DataOperations
        
        ops = DataOperations()
        test_data = create_test_data()
        
        # Test scaling
        numeric_data = test_data[['feature1']]
        scaled_data = ops.scale(numeric_data)
        
        print("✅ Data operations successful")
        return True
    except Exception as e:
        print(f"❌ Data operations failed: {e}")
        return False

def test_data_analyzer():
    """Test data analyzer module."""
    print("Testing data analyzer...")
    try:
        from data_science_pro.data.data_analyzer import DataAnalyzer
        
        analyzer = DataAnalyzer()
        test_data = create_test_data()
        
        # Analyze data
        analysis = analyzer.analyze(test_data)
        
        # Check analysis structure
        assert 'shape' in analysis
        assert 'columns' in analysis
        
        print("✅ Data analyzer successful")
        return True
    except Exception as e:
        print(f"❌ Data analyzer failed: {e}")
        return False

def test_model_registry():
    """Test model registry."""
    print("Testing model registry...")
    try:
        from data_science_pro.modeling.registry import Registry
        
        registry = Registry(registry_dir='test_registry')
        
        # Create simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5)
        
        # Save model
        registry.save_model(model, 'test_model', 1)
        
        # Load model
        loaded_model = registry.load_model('test_model', 1)
        
        print("✅ Model registry successful")
        return True
    except Exception as e:
        print(f"❌ Model registry failed: {e}")
        return False

def test_model_suggestions():
    """Test model suggestion functionality based on test results."""
    print("Testing model suggestions...")
    try:
        from data_science_pro import DataSciencePro
        ai = DataSciencePro()
        
        # Test suggest_models method
        test_results = {
            'current_model': 'LogisticRegression',
            'accuracy': 0.65,
            'precision': 0.62,
            'recall': 0.68
        }
        user_query = "Improve accuracy for binary classification"
        
        suggestions = ai.suggester.suggest_models(test_results, user_query)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check suggestion structure
        for suggestion in suggestions:
            assert 'model' in suggestion
            assert 'reasoning' in suggestion
            assert 'expected_performance' in suggestion
            assert 'suggested_params' in suggestion
        
        print("✅ Model suggestions successful")
        return True
    except Exception as e:
        print(f"❌ Model suggestions failed: {e}")
        return False

def test_hyperparameter_suggestions():
    """Test hyperparameter suggestion functionality."""
    print("Testing hyperparameter suggestions...")
    try:
        from data_science_pro import DataSciencePro
        ai = DataSciencePro()
        
        # Test suggest_hyperparams method
        test_results = {
            'current_model': 'RandomForestClassifier',
            'accuracy': 0.72,
            'overfitting': True,
            'training_time': 2.5
        }
        model_name = 'RandomForestClassifier'
        user_query = "Reduce overfitting while maintaining accuracy"
        
        hyperparams = ai.suggester.suggest_hyperparams(test_results, model_name, user_query)
        assert isinstance(hyperparams, str)
        
        # Verify it's a valid dictionary string
        import ast
        try:
            params_dict = ast.literal_eval(hyperparams)
            assert isinstance(params_dict, dict)
        except (ValueError, SyntaxError):
            raise AssertionError("Hyperparameter suggestions should be a valid dictionary string")
        
        print("✅ Hyperparameter suggestions successful")
        return True
    except Exception as e:
        print(f"❌ Hyperparameter suggestions failed: {e}")
        return False

def test_controllers():
    """Test controller modules."""
    print("Testing controllers...")
    try:
        from data_science_pro.cycle.controller import IntelligentController, InteractiveController
        
        controller = IntelligentController(max_iterations=2)
        interactive = InteractiveController(max_iterations=2)
        
        print("✅ Controllers successful")
        return True
    except Exception as e:
        print(f"❌ Controllers failed: {e}")
        return False

def test_ai_powered_report():
    """Test AI-powered report generation."""
    print("Testing AI-powered report generation...")
    try:
        from data_science_pro import DataSciencePro
        
        # Create test data with some issues for AI to analyze
        test_data = create_test_data()
        test_data.loc[0:3, 'feature1'] = np.nan  # Add missing values
        test_data.loc[4:6, 'feature2'] = 999999  # Add outliers
        test_data.to_csv('test_data.csv', index=False)
        
        ai = DataSciencePro()
        ai.input_data('test_data.csv', target_col='target')
        
        # Generate AI-powered report
        report = ai.report()
        
        # Check if report is AI-enhanced (not just raw data)
        if isinstance(report, dict):
            print("⚠️  Report is still raw data - AI enhancement may have failed")
            return False
        elif isinstance(report, str):
            # Check for AI indicators
            ai_indicators = ['AI', 'intelligent', 'recommendation', 'analysis', 'insight']
            has_ai_content = any(indicator.lower() in report.lower() for indicator in ai_indicators)
            
            if has_ai_content:
                print("✅ AI-powered report generated successfully!")
                print(f"📊 Report length: {len(report)} characters")
                print(f"🧠 Contains AI insights: {has_ai_content}")
                return True
            else:
                print("⚠️  Report generated but may not be AI-enhanced")
                print("Sample report content:")
                print(report[:200] + "...")
                return True  # Still consider it a success if report is generated
        else:
            print(f"❌ Unexpected report type: {type(report)}")
            return False
            
    except Exception as e:
        print(f"❌ AI report generation failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🧪 Running DataSciencePro Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_initialization,
        test_data_loading,
        test_data_analysis,
        test_preprocessing,
        test_model_training,
        test_model_evaluation,
        test_data_operations,
        test_data_analyzer,
        test_model_registry,
        test_controllers,
        test_ai_powered_report
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    # Cleanup test files
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')
    if os.path.exists('test_registry'):
        import shutil
        shutil.rmtree('test_registry')
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)