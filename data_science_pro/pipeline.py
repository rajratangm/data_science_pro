from data_science_pro.api.llm_connector import LLMConnector
from data_science_pro.data.data_loader import DataLoader
from data_science_pro.data.data_analyzer import DataAnalyzer
from data_science_pro.cycle.suggester import Suggester
from data_science_pro.data.data_operations import DataOperations
from data_science_pro.modeling.trainer import Trainer
from data_science_pro.modeling.evaluator import Evaluator
from data_science_pro.modeling.registry import Registry
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class DataSciencePro:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.llm = None
        self.memory = None
        self.data = None
        self.target_col = None
        self.analyzer = DataAnalyzer()
        self.suggester = Suggester(api_key=api_key)  # API key is optional
        self.operations = DataOperations()
        self.model_plan = None
        self.model_instance = None  # Renamed to avoid conflict with method
        self.trainer = Trainer()
        self.evaluator = Evaluator()
        self.registry = Registry()
        self.controller = None  # Will be initialized when api() is called
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def api(self, api_key: str = None):
        """Initializes LangChain LLM + memory."""
        if api_key:
            self.llm = LLMConnector(api_key=api_key)
        self.memory = []
        # Initialize controller with API key for enhanced workflows
        from data_science_pro.cycle.controller import IntelligentController
        self.controller = IntelligentController()

    def input_data(self, file_path, target_col):
        """Loads dataset via DataLoader."""
        loader = DataLoader()
        self.data = loader.load(file_path)
        self.target_col = target_col

    def report(self):
        """Generates AI-powered dynamic analysis reports."""
        analysis_results = self.analyzer.analyze(self.data)
        
        # Use AI-powered report generator for intelligent insights
        try:
            from .cycle.reporter import ReportGenerator
            reporter = ReportGenerator(api_key=self.api_key)
            
            # Get model performance if available
            model_performance = None
            if self.model_instance is not None and self.X_test is not None:
                try:
                    model_performance = self.evaluate()
                except:
                    pass
            
            # Generate AI-powered report
            ai_report = reporter.generate_report(analysis_results, model_performance)
            return ai_report
            
        except Exception as e:
            # Fallback to basic analysis with a warning
            print(f"⚠️  AI report generation failed: {e}")
            print("📊 Falling back to basic analysis...")
            return analysis_results

    def suggestions(self, user_query=None, metrics=None, interactive=True, report=None):
        """
        Advanced AI-powered suggestions with chain-of-thought reasoning and engaging interaction.
        
        Args:
            user_query: User's goal or question
            metrics: Current performance metrics
            interactive: Whether to provide interactive, engaging output
        """
        print("🤖 **AI DATA SCIENTIST ANALYSIS** 🤖")
        print("=" * 50)
        
        # Get comprehensive analysis
        if report is not None:
            analyzer_result = report
        analyzer_result = self.report()
        
        # Handle case where report() returns a string (fallback mode)
        if analyzer_result is None:
            print("⚠️  Running in basic mode due to AI initialization issues.")
            print("📊 Using raw data analysis for suggestions...")
            # Convert back to basic analysis for suggester
            analyzer_result = self.analyzer.analyze(self.data)
        
        # if user_query is None and interactive:
        #     print("\n🎯 What would you like to achieve with your data?")
        #     print("💡 Examples: 'I want to improve accuracy', 'Reduce overfitting', 'Handle missing values better'")
        #     user_query = input("📝 Your goal: ").strip()
            
        #     if not user_query:
        #         user_query = "improve model performance and data quality"
        #         print(f"🔄 Using default goal: {user_query}")

        # Get sophisticated chain-of-thought suggestions with comprehensive CSV analysis
        suggestion_response = self.suggester.suggest_next_action(
            analyzer_result, 
            user_query, 
            metrics,
            csv_data=self.data
        )
        
        if interactive:
            self._display_engaging_suggestions(suggestion_response)
        
        return suggestion_response

    def interactive_workflow(self, workflow_type='intelligent', metric_goal=None, target_metric='f1_score'):
        """
        Run an interactive workflow with enhanced agentic capabilities.
        
        Args:
            workflow_type: Type of workflow ('intelligent', 'preprocessing', 'training', 'testing')
            metric_goal: Target metric value to achieve
            target_metric: Which metric to optimize
        """
        print("🚀 **INTERACTIVE DATA SCIENCE WORKFLOW** 🚀")
        print("="*60)
        print("Welcome to the enhanced data science experience! 🧠✨")
        print("I'll guide you through preprocessing, training, and testing with intelligent suggestions.")
        print()
        
        if workflow_type == 'intelligent':
            # Run the complete intelligent workflow
            print("🧠 **Starting Intelligent Workflow Controller...**")
            print("This will automatically adapt strategies based on your data and goals.")
            print()
            
            final_metric = self.controller.run_intelligent_workflow(
                self, metric_goal=metric_goal, target_metric=target_metric
            )
            
            print(f"\n🎉 **Intelligent workflow completed!**")
            print(f"🏆 Best {target_metric} achieved: {final_metric:.4f}")
            
        elif workflow_type == 'preprocessing':
            print("📋 **Starting Interactive Preprocessing Cycle...**")
            self.controller.run_preprocessing_cycle(self, metric_goal=metric_goal)
            
        elif workflow_type == 'training':
            print("🏋️‍♂️ **Starting Interactive Training Cycle...**")
            self.controller.run_training_cycle(self, metric_goal=metric_goal)
            
        elif workflow_type == 'testing':
            print("🔍 **Starting Interactive Testing Cycle...**")
            self.controller.run_testing_cycle(self, metric_goal=metric_goal)
        
        else:
            print("❌ Invalid workflow type. Choose from: intelligent, preprocessing, training, testing")
            return
        
        print(f"\n✨ **Interactive workflow completed successfully!** ✨")
        print("🎯 Your data science pipeline is now optimized and ready!")
    
    def _display_engaging_suggestions(self, response: dict):
        """Display AI suggestions in an engaging, user-friendly format."""
        
        # Display comprehensive CSV analysis first
        csv_analysis = response.get('csv_analysis', {})
        if csv_analysis and isinstance(csv_analysis, dict):
            print(f"\n📊 **COMPREHENSIVE DATA ANALYSIS:**")
            print(f"📈 **Data Quality Score:** {csv_analysis.get('data_quality_score', 'N/A')}/10")
            print(f"🎯 **Modeling Readiness:** {csv_analysis.get('modeling_readiness', 'N/A')}/10")
            
            if 'critical_issues' in csv_analysis and isinstance(csv_analysis['critical_issues'], list) and len(csv_analysis['critical_issues']) > 0:
                print(f"\n⚠️  **CRITICAL ISSUES IDENTIFIED:**")
                for issue in csv_analysis['critical_issues']:
                    print(f"   • {issue}")
            
            if 'key_insights' in csv_analysis and isinstance(csv_analysis['key_insights'], list) and len(csv_analysis['key_insights']) > 0:
                print(f"\n🔍 **KEY DATA INSIGHTS:**")
                for insight in csv_analysis['key_insights']:
                    print(f"   • {insight}")
            
            if 'preprocessing_recommendations' in csv_analysis and isinstance(csv_analysis['preprocessing_recommendations'], list) and len(csv_analysis['preprocessing_recommendations']) > 0:
                print(f"\n🛠️  **RECOMMENDED PREPROCESSING:**")
                for rec in csv_analysis['preprocessing_recommendations']:
                    print(f"   • {rec}")

        # Display context summary
        context_summary = response.get('context_summary', '')
        if context_summary:
            print(f"\n🧠 **CONTEXTUAL ANALYSIS:**")
            print(f"{context_summary}")
        
        print(f"\n🧠 **STAGE:** {response.get('stage', 'unknown').replace('_', ' ').title()}")
        print(f"📊 **CONFIDENCE:** {response.get('confidence_score', 0):.1%}")
        print("\n" + "="*60)
        
        # Main reasoning with emoji highlights
        reasoning = response.get('reasoning', 'Analysis available - see details above')
        print(f"\n🤔 **MY ANALYSIS:**")
        print(f"{reasoning}")
        
        if 'primary_recommendation' in response:
            print(f"\n🎯 **PRIMARY RECOMMENDATION:**")
            print(f"✅ {response['primary_recommendation']}")
        
        if response.get('alternative_options'):
            print(f"\n🔄 **ALTERNATIVE APPROACHES:**")
            for i, alt in enumerate(response['alternative_options'], 1):
                print(f"   {i}. 💭 {alt}")
        
        if 'implementation_steps' in response:
            print(f"\n📋 **IMPLEMENTATION STEPS:**")
            for i, step in enumerate(response['implementation_steps'], 1):
                print(f"   {i}. ▶️ {step}")
        
        if 'expected_outcomes' in response:
            print(f"\n🚀 **EXPECTED OUTCOMES:**")
            print(f"🎉 {response['expected_outcomes']}")
        
        if 'engagement_message' in response:
            print(f"\n💬 **ENGAGEMENT MESSAGE:**")
            print(f"{response['engagement_message']}")
        
        if 'next_actions' in response:
            print(f"\n⏭️  **NEXT POSSIBLE ACTIONS:**")
            for i, action in enumerate(response['next_actions'], 1):
                print(f"   {i}. 🎯 {action}")
        
        print("\n" + "="*60)
        print("🚀 Ready to take action? Choose your next step!")
    
    def apply_action(self, action_id):
        """Applies preprocessing operation using DataOperations."""
        # Use DataOperations for all preprocessing
        if action_id == 'drop_na':
            self.data = self.data.dropna()
        elif action_id == 'drop_duplicates':
            self.data = self.data.drop_duplicates()
        elif action_id == 'drop_constant':
            constant_cols = [col for col in self.data.columns if self.data[col].nunique() == 1]
            self.data = self.operations.drop_columns(self.data, constant_cols)
        elif action_id == 'drop_high_na':
            thresh = len(self.data) * 0.5
            high_na_cols = [col for col in self.data.columns if col != self.target_col and self.data[col].isna().sum() > thresh]
            self.data = self.operations.drop_columns(self.data, high_na_cols)
        elif action_id == 'fill_na':
            for col in self.data.columns:
                if self.data[col].dtype in ['float64', 'int64']:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                else:
                    mode_values = self.data[col].mode()
                    if len(mode_values) > 0:
                        self.data[col] = self.data[col].fillna(mode_values[0])
                    else:
                        self.data[col] = self.data[col].fillna('unknown')
        elif action_id == 'encode_categorical':
            # One-hot encode all object columns except target
            cols = [col for col in self.data.select_dtypes(include='object').columns if col != self.target_col]
            self.data = self.operations.encode(self.data, columns=cols)
        elif action_id == 'scale_numeric':
            num_cols = [col for col in self.data.select_dtypes(include='number').columns if col != self.target_col]
            self.data = self.operations.scale(self.data, columns=num_cols, method='standard')
        elif action_id == 'feature_gen':
            self.data = self.operations.generate_features(self.data)
        # After preprocessing, drop high cardinality columns and non-numeric columns
        if hasattr(self, 'target_col') and self.target_col in self.data.columns:
            high_card_cols = [col for col in self.data.columns if self.data[col].nunique() > 50 and col != self.target_col]
            self.data = self.operations.drop_columns(self.data, high_card_cols)
            non_numeric_cols = [col for col in self.data.columns if col != self.target_col and not pd.api.types.is_numeric_dtype(self.data[col])]
            self.data = self.operations.drop_columns(self.data, non_numeric_cols)
            # Train/test split
            X = self.data.drop(columns=[self.target_col])
            y = self.data[self.target_col]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def set_model(self, model_name, hyperparams):
        """Instantiate and set the model with user-selected hyperparameters."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        # Add more models as needed
        if model_name.lower() == 'randomforest':
            self.model_instance = RandomForestClassifier(**hyperparams)
        elif model_name.lower() == 'logisticregression':
            self.model_instance = LogisticRegression(**hyperparams)
        else:
            print(f"Model {model_name} not recognized. Using LogisticRegression by default.")
            self.model_instance = LogisticRegression(**hyperparams)

    def train(self):
        """Trains ML model."""
        if self.model_instance is not None and self.X_train is not None and self.y_train is not None:
            self.model_instance.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluate the trained model with comprehensive metrics.
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model_instance is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test data available. Ensure data is loaded and split.")
        
        # Make predictions
        y_pred = self.model_instance.predict(self.X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(self.model_instance, 'predict_proba'):
            try:
                y_pred_proba = self.model_instance.predict_proba(self.X_test)
            except:
                pass
        
        # Calculate comprehensive metrics
        metrics = self.evaluator.evaluate(self.y_test, y_pred, y_pred_proba)
        
        return metrics

    def cross_validate(self, cv_folds=5):
        """
        Perform cross-validation with comprehensive metrics.
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Cross-validation metrics
        """
        if self.model_instance is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data available. Ensure data is loaded and split.")
        
        from sklearn.model_selection import cross_validate, StratifiedKFold
        import numpy as np
        
        # Set up cross-validation
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Define scoring metrics
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        
        # Perform cross-validation
        cv_results = cross_validate(
            self.model_instance, self.X_train, self.y_train, 
            cv=cv_strategy, scoring=scoring, 
            return_train_score=True
        )
        
        # Calculate comprehensive CV metrics
        cv_metrics = {
            'cv_folds': cv_folds,
            'mean_accuracy': float(np.mean(cv_results['test_accuracy'])),
            'std_accuracy': float(np.std(cv_results['test_accuracy'])),
            'mean_precision': float(np.mean(cv_results['test_precision'])),
            'std_precision': float(np.std(cv_results['test_precision'])),
            'mean_recall': float(np.mean(cv_results['test_recall'])),
            'std_recall': float(np.std(cv_results['test_recall'])),
            'mean_f1': float(np.mean(cv_results['test_f1'])),
            'std_f1': float(np.std(cv_results['test_f1'])),
            'train_accuracy': float(np.mean(cv_results['train_accuracy'])),
            'train_std': float(np.std(cv_results['train_accuracy'])),
            'overfitting_score': float(np.mean(cv_results['train_accuracy']) - np.mean(cv_results['test_accuracy'])),
            'stability_score': float(1 - (np.std(cv_results['test_accuracy']) / np.mean(cv_results['test_accuracy']))),
            'all_scores': {
                'test_accuracy': cv_results['test_accuracy'].tolist(),
                'test_precision': cv_results['test_precision'].tolist(),
                'test_recall': cv_results['test_recall'].tolist(),
                'test_f1': cv_results['test_f1'].tolist()
            }
        }
        
        return cv_metrics

    def test(self):
        """Tests ML model (could be same as evaluate or extended)."""
        return self.evaluate()

    def save_model(self, path):
        """Saves trained model."""
        if self.model_instance is not None:
            self.registry.save_model(self.model_instance, 'model', 1)  # Versioning can be improved

    def run_full_cycle(self, controller, metric_goal=None):
        """
        Run full cyclic workflow: preprocessing, training, testing until metrics are satisfied.
        """
        print("Starting preprocessing cycle...")
        controller.run_preprocessing_cycle(self, metric_goal=metric_goal)
        print("Starting training cycle...")
        controller.run_training_cycle(self, metric_goal=metric_goal)
        print("Starting testing cycle...")
        controller.run_testing_cycle(self, metric_goal=metric_goal)
