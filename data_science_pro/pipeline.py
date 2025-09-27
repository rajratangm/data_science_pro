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
            self.llm = LLMConnector(api_key)
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
            reporter = ReportGenerator()
            
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

    def suggestions(self, user_query=None, metrics=None, interactive=True):
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
        analyzer_result = self.report()
        
        # Handle case where report() returns a string (fallback mode)
        if isinstance(analyzer_result, str):
            print("⚠️  Running in basic mode due to AI initialization issues.")
            print("📊 Using raw data analysis for suggestions...")
            # Convert back to basic analysis for suggester
            analyzer_result = self.analyzer.analyze(self.data)
        
        if user_query is None and interactive:
            print("\n🎯 What would you like to achieve with your data?")
            print("💡 Examples: 'I want to improve accuracy', 'Reduce overfitting', 'Handle missing values better'")
            user_query = input("📝 Your goal: ").strip()
            
            if not user_query:
                user_query = "improve model performance and data quality"
                print(f"🔄 Using default goal: {user_query}")
        
        # Get sophisticated chain-of-thought suggestions
        suggestion_response = self.suggester.suggest_next_action(analyzer_result, user_query, metrics)
        
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
        
        print(f"\n🧠 **STAGE:** {response['stage'].replace('_', ' ').title()}")
        print(f"📊 **CONFIDENCE:** {response['confidence_score']:.1%}")
        print("\n" + "="*60)
        
        # Main reasoning with emoji highlights
        reasoning = response['reasoning']
        print(f"\n🤔 **MY ANALYSIS:**")
        print(f"{reasoning}")
        
        print(f"\n🎯 **PRIMARY RECOMMENDATION:**")
        print(f"✅ {response['primary_recommendation']}")
        
        if response['alternative_options']:
            print(f"\n🔄 **ALTERNATIVE APPROACHES:**")
            for i, alt in enumerate(response['alternative_options'], 1):
                print(f"   {i}. 💭 {alt}")
        
        print(f"\n📋 **IMPLEMENTATION STEPS:**")
        for i, step in enumerate(response['implementation_steps'], 1):
            print(f"   {i}. ▶️ {step}")
        
        print(f"\n🚀 **EXPECTED OUTCOMES:**")
        print(f"🎉 {response['expected_outcomes']}")
        
        print(f"\n💬 **ENGAGEMENT MESSAGE:**")
        print(f"{response['engagement_message']}")
        
        print(f"\n⏭️  **NEXT POSSIBLE ACTIONS:**")
        for i, action in enumerate(response['next_actions'], 1):
            print(f"   {i}. 🎯 {action}")
        
        print("\n" + "="*60)
        print("🚀 Ready to take action? Choose your next step!")
    
    def interactive_workflow(self, target_metric='accuracy', target_value=0.85):
        """
        Fully interactive workflow that guides user through complete data science process.
        
        Args:
            target_metric: Which metric to optimize ('accuracy', 'precision', 'recall', 'f1')
            target_value: Target value for the metric
        """
        print("🎉 **WELCOME TO INTERACTIVE DATA SCIENCE WORKFLOW!** 🎉")
        print("="*60)
        print("I'm your AI data scientist assistant. Let's build an amazing model together! 🤝")
        print()
        
        step = 1
        while True:
            print(f"\n📍 **STEP {step}: Current Status Check**")
            print("-" * 40)
            
            # Get current metrics if model exists
            current_metrics = None
            if self.model_instance is not None and self.X_test is not None:
                try:
                    current_metrics = self.evaluate()
                    print(f"📊 Current Performance:")
                    for metric, value in current_metrics.items():
                        print(f"   • {metric.title()}: {value:.4f}")
                    
                    # Check if target is achieved
                    if target_metric in current_metrics:
                        if current_metrics[target_metric] >= target_value:
                            print(f"\n🎊 **CONGRATULATIONS! Target achieved!** 🎊")
                            print(f"   {target_metric.title()}: {current_metrics[target_metric]:.4f} >= {target_value}")
                            break
                        else:
                            print(f"\n🎯 Target: {target_metric.title()} >= {target_value}")
                            print(f"📈 Current: {current_metrics[target_metric]:.4f} - Keep optimizing!")
                except:
                    pass
            
            # Get AI suggestions
            user_goal = input(f"\n🎯 What's your main goal right now? (or press Enter for AI suggestions): ").strip()
            if not user_goal:
                user_goal = f"improve {target_metric} to reach {target_value}"
            
            print(f"\n🤖 **Getting AI recommendations...**")
            suggestions = self.suggestions(user_query=user_goal, metrics=current_metrics, interactive=True)
            
            # Let user choose next action
            print(f"\n🎮 **Choose your next action:**")
            available_actions = response['next_actions'] if 'response' in locals() else suggestions['next_actions']
            
            for i, action in enumerate(available_actions, 1):
                print(f"   {i}. ⚡ {action}")
            print(f"   {len(available_actions)+1}. 🔄 Get different suggestions")
            print(f"   {len(available_actions)+2}. 🛑 Finish workflow")
            
            choice = input(f"\n📝 Your choice (1-{len(available_actions)+2}): ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_actions):
                    selected_action = available_actions[choice_num-1]
                    print(f"\n🚀 **Executing: {selected_action}**")
                    
                    if selected_action in ['randomforest', 'logisticregression']:
                        # Model selection
                        if selected_action == 'randomforest':
                            self.set_model('randomforest', {'n_estimators': 200, 'max_depth': 15})
                        else:
                            self.set_model('logisticregression', {'C': 1.0, 'max_iter': 1000})
                        print("✅ Model configured!")
                        
                    elif selected_action in ['drop_na', 'fill_na', 'encode_categorical', 'scale_numeric', 'feature_gen']:
                        # Preprocessing action
                        self.apply_action(selected_action)
                        print(f"✅ Action '{selected_action}' completed!")
                        print(f"📊 New data shape: {self.data.shape}")
                        
                    elif selected_action == 'train':
                        # Training
                        print("🏋️‍♂️ Training model...")
                        self.train()
                        print("✅ Training completed!")
                        
                    elif selected_action == 'evaluate':
                        # Evaluation
                        results = self.evaluate()
                        print("📊 Evaluation Results:")
                        for metric, value in results.items():
                            print(f"   • {metric.title()}: {value:.4f}")
                    
                elif choice_num == len(available_actions)+1:
                    print("🔄 Getting new suggestions...")
                    continue
                elif choice_num == len(available_actions)+2:
                    print("\n🎉 **Workflow completed!** Great job! 🎉")
                    break
            
            step += 1
            
            # Prevent infinite loops
            if step > 20:
                print("\n⚠️  Maximum steps reached. Consider adjusting your target or approach.")
                break
        
        # Final summary
        print(f"\n📋 **WORKFLOW SUMMARY:**")
        print(f"🎯 Target: {target_metric} >= {target_value}")
        if current_metrics and target_metric in current_metrics:
            print(f"🏆 Final {target_metric.title()}: {current_metrics[target_metric]:.4f}")
            if current_metrics[target_metric] >= target_value:
                print("🎊 **TARGET ACHIEVED!** 🎊")
            else:
                print("📈 Good progress! Consider further optimization.")
        print("\n✨ **Thank you for using AI-Powered Data Science!** ✨")

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
        """Evaluates ML model."""
        if self.model_instance is not None and self.X_test is not None and self.y_test is not None:
            try:
                y_pred = self.model_instance.predict(self.X_test)
                if y_pred is None:
                    raise ValueError("Model prediction returned None")
                return self.evaluator.evaluate(self.y_test, y_pred)
            except Exception as e:
                print(f"❌ Evaluation error: {e}")
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
        else:
            print("⚠️  Cannot evaluate: Model or test data not available")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

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
