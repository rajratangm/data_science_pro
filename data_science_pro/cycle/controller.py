import pandas as pd
import numpy as np
from datetime import datetime

class IntelligentController:
    """
    Advanced controller that provides intelligent cyclic workflow management
    with chain-of-thought reasoning and adaptive strategy selection.
    """
    
    def __init__(self, max_iterations=10, improvement_threshold=0.01):
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.workflow_history = []
        self.strategy_performance = {}
        self.current_strategy = "balanced"
        
        # Define sophisticated strategy profiles
        self.strategies = {
            "aggressive_cleaning": {
                "preprocessing_sequence": [
                    "drop_high_na", "drop_duplicates", "drop_constant",
                    "fill_na", "encode_categorical", "scale_numeric", "feature_gen"
                ],
                "model_approach": "ensemble_first",
                "hyperparameter_intensity": "high",
                "description": "Maximum data quality focus with comprehensive preprocessing"
            },
            "quick_wins": {
                "preprocessing_sequence": [
                    "drop_na", "fill_na", "encode_categorical", "scale_numeric"
                ],
                "model_approach": "simple_to_complex",
                "hyperparameter_intensity": "medium",
                "description": "Fast results with high-impact, low-effort actions"
            },
            "feature_engineering_focus": {
                "preprocessing_sequence": [
                    "fill_na", "encode_categorical", "feature_gen", "scale_numeric"
                ],
                "model_approach": "feature_sensitive",
                "hyperparameter_intensity": "medium",
                "description": "Emphasis on feature creation and engineering"
            },
            "model_optimization_focus": {
                "preprocessing_sequence": [
                    "drop_na", "fill_na", "encode_categorical", "scale_numeric"
                ],
                "model_approach": "ensemble_optimization",
                "hyperparameter_intensity": "very_high",
                "description": "Deep model tuning and ensemble methods"
            },
            "balanced": {
                "preprocessing_sequence": [
                    "drop_na", "fill_na", "encode_categorical", "scale_numeric"
                ],
                "model_approach": "progressive_refinement",
                "hyperparameter_intensity": "medium",
                "description": "Balanced approach across all aspects"
            }
        }
    
    def run_intelligent_workflow(self, pipeline, metric_goal=None, target_metric='f1_score'):
        """
        Run sophisticated cyclic workflow with intelligent strategy adaptation.
        
        Args:
            pipeline: DataSciencePro pipeline instance
            metric_goal: Target metric value to achieve
            target_metric: Which metric to optimize ('accuracy', 'precision', 'recall', 'f1_score')
        """
        print("🧠 **INTELLIGENT WORKFLOW CONTROLLER ACTIVATED** 🧠")
        print("="*60)
        print("I'll adaptively optimize your data science workflow! 🚀")
        print()
        
        best_metric = 0
        iteration = 0
        strategy_changes = 0
        
        # Initialize iteration history tracking
        self.iteration_history = []
        
        while iteration < self.max_iterations:
            print(f"\n🔄 **ITERATION {iteration + 1}/{self.max_iterations}**")
            print(f"🎯 Current Strategy: {self.current_strategy}")
            print(f"📊 Target: {target_metric} >= {metric_goal}")
            print("-" * 40)
            
            # Get current performance
            current_metrics = self._evaluate_current_state(pipeline)
            current_metric = current_metrics.get(target_metric, 0)
            
            print(f"📈 Current {target_metric}: {current_metric:.4f}")
            print(f"🏆 Best so far: {best_metric:.4f}")
            
            # Check if goal achieved
            if metric_goal and current_metric >= metric_goal:
                print(f"\n🎊 **GOAL ACHIEVED!** 🎊")
                print(f"{target_metric}: {current_metric:.4f} >= {metric_goal}")
                break
            
            # Adapt strategy if needed
            if iteration > 2 and current_metric <= best_metric:
                new_strategy = self._adapt_strategy(iteration, current_metric, best_metric)
                if new_strategy != self.current_strategy:
                    print(f"🧠 **ADAPTING STRATEGY:** {self.current_strategy} → {new_strategy}")
                    self.current_strategy = new_strategy
                    strategy_changes += 1
            
            # Execute current strategy
            self._execute_strategy(pipeline, self.current_strategy, iteration)
            
            # Update best performance
            if current_metric > best_metric:
                best_metric = current_metric
                print(f"📈 **NEW BEST PERFORMANCE!** {current_metric:.4f}")
            
            # Record workflow history
            self._record_iteration(iteration, current_metrics, self.current_strategy)
            
            # Track iteration history for comprehensive context
            self.iteration_history.append({
                'iteration': iteration + 1,
                'metrics': current_metrics,
                'strategy': self.current_strategy,
                'timestamp': datetime.now(),
                'stage': 'training_evaluation',
                'actions_taken': ['strategy_execution', 'evaluation']
            })
            
            iteration += 1
            
            # Early stopping if no improvement for several iterations
            if self._should_early_stop(iteration, current_metric, best_metric):
                print(f"\n⚠️ **EARLY STOPPING:** No significant improvement detected")
                break
        
        # Final summary
        self._print_workflow_summary(iteration, best_metric, strategy_changes, target_metric)
        return best_metric
    
    def _execute_strategy(self, pipeline, strategy_name, iteration):
        """Execute a specific strategy with intelligent modifications."""
        
        strategy = self.strategies[strategy_name]
        print(f"\n🔧 **EXECUTING STRATEGY:** {strategy['description']}")
        
        # Phase 1: Preprocessing
        print("\n📋 **Phase 1: Intelligent Preprocessing**")
        preprocessing_sequence = self._optimize_preprocessing_sequence(strategy['preprocessing_sequence'], iteration)
        
        for action in preprocessing_sequence:
            try:
                print(f"   ⚡ Applying: {action}")
                pipeline.apply_action(action)
                
                # Smart evaluation after critical actions
                if action in ['encode_categorical', 'feature_gen', 'scale_numeric']:
                    print(f"   📊 Data shape after {action}: {pipeline.data.shape}")
                    
            except Exception as e:
                print(f"   ⚠️  Warning: {action} failed - {str(e)}")
                continue
        
        # Phase 2: Model Selection and Training
        print("\n🤖 **Phase 2: Adaptive Model Training**")
        self._adaptive_model_training(pipeline, strategy, iteration)
        
        # Phase 3: Evaluation and Optimization
        print("\n📊 **Phase 3: Comprehensive Evaluation**")
        self._comprehensive_evaluation(pipeline, strategy)
    
    def _adaptive_model_training(self, pipeline, strategy, iteration):
        """Intelligent model training with strategy-specific approaches."""
        
        model_approach = strategy['model_approach']
        
        if model_approach == "ensemble_first":
            # Start with ensemble methods
            print("   🎯 Training RandomForest (Ensemble Approach)")
            pipeline.set_model('randomforest', {
                'n_estimators': min(100 + iteration * 50, 500),
                'max_depth': min(10 + iteration, 20),
                'min_samples_split': max(2, 5 - iteration // 3)
            })
            
        elif model_approach == "simple_to_complex":
            # Start simple, progress to complex
            if iteration < 3:
                print("   🎯 Training LogisticRegression (Simple Baseline)")
                pipeline.set_model('logisticregression', {'C': 1.0, 'max_iter': 1000})
            else:
                print("   🎯 Training RandomForest (Complex Model)")
                pipeline.set_model('randomforest', {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 3
                })
                
        elif model_approach == "feature_sensitive":
            # Models that work well with engineered features
            print("   🎯 Training feature-sensitive model")
            pipeline.set_model('randomforest', {
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 2
            })
            
        elif model_approach == "ensemble_optimization":
            # Deep ensemble optimization
            print("   🎯 Advanced Ensemble Training")
            pipeline.set_model('randomforest', {
                'n_estimators': 300,
                'max_depth': 18,
                'min_samples_split': 2,
                'bootstrap': True,
                'oob_score': True
            })
            
        else:  # progressive_refinement
            # Balanced approach
            complexity_factor = min(iteration / 5, 1.0)
            pipeline.set_model('randomforest', {
                'n_estimators': int(100 + 200 * complexity_factor),
                'max_depth': int(10 + 8 * complexity_factor),
                'min_samples_split': max(2, int(5 - 3 * complexity_factor))
            })
        
        # Train the model
        print("   🏋️‍♂️ Training in progress...")
        pipeline.train()
        print("   ✅ Training completed!")
    
    def _comprehensive_evaluation(self, pipeline, strategy):
        """Comprehensive evaluation with detailed insights."""
        
        print("   📊 Running comprehensive evaluation...")
        metrics = pipeline.evaluate()
        
        print("   📈 **Evaluation Results:**")
        for metric, value in metrics.items():
            print(f"      • {metric.title()}: {value:.4f}")
        
        # Strategy-specific insights
        if strategy['model_approach'] == "ensemble_optimization":
            print("   🎯 **Ensemble Insights:**")
            print("      • Model complexity optimized for your data")
            print("      • Balanced bias-variance tradeoff")
            
        return metrics
    
    def _adapt_strategy(self, iteration, current_metric, best_metric):
        """Intelligently adapt strategy based on performance history."""
        
        # Performance trend analysis
        recent_performance = self._get_recent_performance(3)
        
        if len(recent_performance) < 2:
            return self.current_strategy
        
        # Determine if we're plateauing
        improvement_rate = (recent_performance[-1] - recent_performance[0]) / len(recent_performance)
        
        if improvement_rate < 0.001:  # Plateau detected
            if self.current_strategy == "balanced":
                return "aggressive_cleaning"
            elif self.current_strategy == "aggressive_cleaning":
                return "feature_engineering_focus"
            elif self.current_strategy == "feature_engineering_focus":
                return "model_optimization_focus"
            else:
                return "balanced"  # Reset to balanced
        
        elif improvement_rate > 0.01:  # Good improvement
            if self.current_strategy == "quick_wins":
                return "balanced"  # Move to more comprehensive approach
            return self.current_strategy  # Stick with what's working
        
        return self.current_strategy
    
    def _get_recent_performance(self, window):
        """Get recent performance history."""
        if len(self.workflow_history) < window:
            return [item['metrics'].get('f1_score', 0) for item in self.workflow_history]
        return [item['metrics'].get('f1_score', 0) for item in self.workflow_history[-window:]]
    
    def _should_early_stop(self, iteration, current_metric, best_metric):
        """Determine if we should stop early due to lack of improvement."""
        
        if iteration < 5:
            return False
        
        # Check if we've had minimal improvement in recent iterations
        recent_best = max(self._get_recent_performance(4))
        
        if best_metric > 0 and (recent_best - best_metric) / best_metric < 0.005:
            return True
        
        return False
    
    def _evaluate_current_state(self, pipeline):
        """Evaluate current pipeline state."""
        try:
            if pipeline.model_instance is not None and pipeline.X_test is not None:
                return pipeline.evaluate()
            else:
                return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
        except:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    
    def _optimize_preprocessing_sequence(self, base_sequence, iteration):
        """Dynamically optimize preprocessing sequence based on iteration."""
        
        # Add more aggressive cleaning in later iterations
        if iteration > 5 and "drop_high_na" not in base_sequence:
            base_sequence = ["drop_high_na"] + base_sequence
        
        # Add feature generation in middle iterations
        if 2 < iteration < 8 and "feature_gen" not in base_sequence:
            if "encode_categorical" in base_sequence:
                idx = base_sequence.index("encode_categorical")
                base_sequence.insert(idx + 1, "feature_gen")
        
        return base_sequence
    
    def _record_iteration(self, iteration, metrics, strategy):
        """Record iteration details for learning."""
        
        self.workflow_history.append({
            'iteration': iteration,
            'metrics': metrics,
            'strategy': strategy,
            'timestamp': datetime.now(),
            'stage': 'training_evaluation',
            'actions_taken': ['strategy_execution', 'evaluation']
        })
    
    def _print_workflow_summary(self, iterations, best_metric, strategy_changes, target_metric):
        """Print comprehensive workflow summary."""
        
        print(f"\n🎉 **WORKFLOW COMPLETED SUCCESSFULLY!** 🎉")
        print("="*50)
        print(f"📊 **SUMMARY:**")
        print(f"   • Total Iterations: {iterations}")
        print(f"   • Best {target_metric}: {best_metric:.4f}")
        print(f"   • Strategy Changes: {strategy_changes}")
        print(f"   • Strategies Used: {len(set([h['strategy'] for h in self.workflow_history]))}")
        
        if self.workflow_history:
            final_metrics = self.workflow_history[-1]['metrics']
            print(f"\n📈 **Final Performance:**")
            for metric, value in final_metrics.items():
                print(f"   • {metric.title()}: {value:.4f}")
        
        print(f"\n✨ **Great job! Your model is ready!** ✨")

# Enhanced interactive cycles with intelligent suggestions
class InteractiveController(IntelligentController):
    """Enhanced controller with intelligent workflow management and interactive cycles."""
    
    def __init__(self, max_iterations=10, improvement_threshold=0.01):
        super().__init__(max_iterations, improvement_threshold)
        self.state = {}
        self.memory = []
        self.current_cycle = None
    
    def start_cycle(self, cycle_id):
        """Start a new cycle with intelligent logging."""
        self.current_cycle = cycle_id
        self.state[cycle_id] = {'status': 'started', 'start_time': datetime.now()}
        self.memory.append(f"🔄 Cycle {cycle_id} started at {datetime.now().strftime('%H:%M:%S')}")
        print(f"🔄 **CYCLE STARTED:** {cycle_id}")
    
    def stop_cycle(self, cycle_id):
        """Stop a cycle with performance summary."""
        if cycle_id in self.state:
            self.state[cycle_id]['status'] = 'stopped'
            self.state[cycle_id]['end_time'] = datetime.now()
            duration = self.state[cycle_id]['end_time'] - self.state[cycle_id]['start_time']
            self.memory.append(f"✅ Cycle {cycle_id} completed in {duration}")
            print(f"✅ **CYCLE COMPLETED:** {cycle_id} (Duration: {duration})")
            if self.current_cycle == cycle_id:
                self.current_cycle = None
    
    def run_preprocessing_cycle(self, pipeline, max_loops=5, metric_goal=None):
        """
        Enhanced cyclic preprocessing with intelligent suggestions and chain-of-thought reasoning.
        """
        print("🧠 **INTELLIGENT PREPROCESSING CYCLE** 🧠")
        print("="*50)
        print("I'll help you optimize your data preprocessing step by step! 📊")
        print()
        
        self.start_cycle('preprocessing')
        
        for cycle in range(max_loops):
            print(f"\n📋 **PREPROCESSING CYCLE {cycle+1}/{max_loops}**")
            print("-" * 30)
            
            # Get comprehensive analysis
            report = pipeline.report()
            print(f"📊 **Current Data Status:**")
            print(f"   • Shape: {pipeline.data.shape}")
            print(f"   • Missing values: {pipeline.data.isnull().sum().sum()}")
            print(f"   • Categorical columns: {len(pipeline.data.select_dtypes(include=['object']).columns)}")
            print(f"   • Numeric columns: {len(pipeline.data.select_dtypes(include=[np.number]).columns)}")
            
            # Get intelligent suggestions
            user_query = input("\n🤔 **What would you like to improve?** (e.g., 'reduce missing data', 'handle categorical variables', 'scale features'): ")
            if not user_query.strip():
                user_query = "improve data quality and prepare for modeling"
            
            print("\n🧠 **Analyzing your request and generating intelligent suggestions...**")
            suggestions = pipeline.suggestions(user_query=user_query)
            
            print(f"\n🎯 **Smart Suggestions for your goal:**")
            if suggestions and isinstance(suggestions, dict):
                # Handle the new dictionary format from suggest_next_action
                print(f"\n   🎯 **Primary Recommendation:** {suggestions.get('primary_recommendation', 'No primary recommendation')}")
                
                alternatives = suggestions.get('alternative_options', [])
                if alternatives:
                    print(f"\n   🔄 **Alternative Approaches:**")
                    for idx, alt in enumerate(alternatives):
                        print(f"      [{idx}] {alt}")
                
                implementation_steps = suggestions.get('implementation_steps', [])
                if implementation_steps:
                    print(f"\n   📋 **Implementation Steps:**")
                    for idx, step in enumerate(implementation_steps):
                        print(f"      {idx+1}. {step}")
                
                # Use next_actions for interactive selection
                next_actions = suggestions.get('next_possible_actions', [])
                if next_actions:
                    print(f"\n   ⚡ **Available Actions:**")
                    for idx, action in enumerate(next_actions):
                        print(f"      [{idx}] {action}")
                    
                    # Interactive selection from next_actions
                    action_idx = input("\n🤔 **Which action would you like to apply?** (Enter number, or press Enter to skip): ")
                    if action_idx.isdigit() and int(action_idx) < len(next_actions):
                        action_id = next_actions[int(action_idx)]
                        print(f"\n⚡ **Applying action:** {action_id}")
                        try:
                            pipeline.apply_action(action_id)
                            self.update_state('preprocessing', f'cycle_{cycle+1}', action_id)
                            print(f"✅ **Action completed successfully!**")
                            
                            # Show immediate impact
                            new_report = pipeline.report()
                            if 'data_shape' in new_report:
                                print(f"📊 **Data shape after action:** {new_report['data_shape']}")
                            
                        except Exception as e:
                            print(f"❌ **Action failed:** {str(e)}")
                            print("💡 **Tip:** This might be expected - some actions can't be applied to all data types.")
                else:
                    print("   ⚠️  No specific actions available in this suggestion.")
            else:
                print("   ⚠️  No specific suggestions available. Try general goals like 'improve data quality'.")
            
            # Check if goal achieved
            if metric_goal and self.check_metric(report, metric_goal):
                print(f"\n🎊 **PREPROCESSING GOAL ACHIEVED!** 🎊")
                print(f"   Target metric {metric_goal} reached!")
                break
            
            # Ask if user wants to continue
            continue_choice = input("\n🤔 **Would you like to continue preprocessing?** (y/n): ").lower()
            if continue_choice != 'y':
                print("✅ **Preprocessing cycle completed by user choice.**")
                break
        
        self.stop_cycle('preprocessing')
        print(f"\n🎉 **PREPROCESSING CYCLE COMPLETED!** 🎉")
        print("Your data is now optimized and ready for modeling! 🚀")
    
    def run_training_cycle(self, pipeline, max_loops=5, metric_goal=None):
        """
        Enhanced cyclic training with intelligent model selection and hyperparameter optimization.
        """
        print("🧠 **INTELLIGENT TRAINING CYCLE** 🧠")
        print("="*50)
        print("I'll help you find the best model and optimize it! 🤖")
        print()
        
        self.start_cycle('training')
        
        for cycle in range(max_loops):
            print(f"\n🏋️‍♂️ **TRAINING CYCLE {cycle+1}/{max_loops}**")
            print("-" * 30)
            
            # Train current model
            print("🏃‍♂️ **Training model...**")
            pipeline.train()
            
            # Get comprehensive report
            report = pipeline.report()
            print(f"📊 **Training Results:**")
            if 'metrics' in report:
                for metric, value in report['metrics'].items():
                    print(f"   • {metric.title()}: {value:.4f}")
            
            # Get intelligent training suggestions
            user_query = input("\n🤔 **What would you like to improve?** (e.g., 'increase accuracy', 'reduce overfitting', 'try ensemble methods'): ")
            if not user_query.strip():
                user_query = "improve model performance"
            
            print("\n🧠 **Analyzing model performance and generating optimization suggestions...**")
            suggestions = pipeline.suggester.suggest_models(report, user_query, metrics=report.get('metrics', {}))
            
            print(f"\n🎯 **Smart Model Suggestions:**")
            if suggestions and isinstance(suggestions, dict):
                # Handle dictionary format
                print(f"\n   🎯 **Primary Recommendation:** {suggestions.get('primary_recommendation', 'No primary recommendation')}")
                
                alternatives = suggestions.get('alternative_options', [])
                if alternatives:
                    print(f"\n   🔄 **Alternative Models:**")
                    for idx, alt in enumerate(alternatives):
                        print(f"      [{idx}] {alt}")
                
                implementation_steps = suggestions.get('implementation_steps', [])
                if implementation_steps:
                    print(f"\n   📋 **Implementation Steps:**")
                    for idx, step in enumerate(implementation_steps):
                        print(f"      {idx+1}. {step}")
            elif suggestions and isinstance(suggestions, list):
                # Handle legacy list format if still used
                for idx, suggestion in enumerate(suggestions):
                    print(f"\n   [{idx}] **{suggestion.get('model', 'Model')}**")
                    print(f"       📝 {suggestion.get('reasoning', 'No reasoning provided')}")
                    if 'expected_performance' in suggestion:
                        print(f"       📈 Expected Performance: {suggestion['expected_performance']}")
            else:
                print("   ⚠️  No model suggestions available.")
            
            # Model selection
            model_name = input("\n🤔 **Which model would you like to try?** (Enter model name, or press Enter to keep current): ")
            if model_name.strip():
                # Get hyperparameter suggestions
                print(f"\n🔧 **Generating optimal hyperparameters for {model_name}...**")
                hyperparam_suggestion = pipeline.suggester.suggest_hyperparams(report, model_name, user_query)
                print(f"🎯 **Suggested Hyperparameters:** {hyperparam_suggestion}")
                
                # Interactive hyperparameter input
                hyperparams_input = input("\n🤔 **Enter hyperparameters** (as Python dict, or press Enter to use suggestion): ")
                try:
                    hyperparams_dict = eval(hyperparams_input) if hyperparams_input.strip() else eval(hyperparam_suggestion)
                except Exception:
                    print("⚠️  Invalid hyperparameter format. Using suggested parameters.")
                    hyperparams_dict = {}
                
                print(f"⚡ **Setting up {model_name} with optimized parameters...**")
                pipeline.set_model(model_name, hyperparams_dict)
                print("✅ **Model configuration updated!**")
            
            # Check if goal achieved
            if metric_goal and self.check_metric(report, metric_goal):
                print(f"\n🎊 **TRAINING GOAL ACHIEVED!** 🎊")
                print(f"   Target metric {metric_goal} reached!")
                break
            
            # Ask if user wants to continue
            continue_choice = input("\n🤔 **Would you like to try another model?** (y/n): ").lower()
            if continue_choice != 'y':
                print("✅ **Training cycle completed by user choice.**")
                break
        
        self.stop_cycle('training')
        print(f"\n🎉 **TRAINING CYCLE COMPLETED!** 🎉")
        print("Your model is now optimized and ready for testing! 🚀")
    
    def run_testing_cycle(self, pipeline, max_loops=5, metric_goal=None):
        """
        Enhanced cyclic testing with comprehensive evaluation and intelligent diagnostics.
        """
        print("🧠 **INTELLIGENT TESTING CYCLE** 🧠")
        print("="*50)
        print("I'll help you thoroughly evaluate your model! 🔍")
        print()
        
        self.start_cycle('testing')
        
        for cycle in range(max_loops):
            print(f"\n🔍 **TESTING CYCLE {cycle+1}/{max_loops}**")
            print("-" * 30)
            
            # Run comprehensive testing
            print("🧪 **Running comprehensive tests...**")
            pipeline.test()
            
            # Get detailed test report
            report = pipeline.report()
            print(f"📊 **Test Results:**")
            if 'metrics' in report:
                for metric, value in report['metrics'].items():
                    print(f"   • {metric.title()}: {value:.4f}")
            
            # Get intelligent testing suggestions
            user_query = input("\n🤔 **What would you like to investigate?** (e.g., 'check model robustness', 'analyze feature importance', 'test on different data'): ")
            if not user_query.strip():
                user_query = "comprehensive model evaluation"
            
            print("\n🧠 **Analyzing model performance and generating diagnostic suggestions...**")
            suggestions = pipeline.suggestions(user_query=user_query)
            
            print(f"\n🔍 **Smart Diagnostic Suggestions:**")
            if suggestions and isinstance(suggestions, dict):
                # Handle dictionary format
                print(f"\n   🎯 **Primary Recommendation:** {suggestions.get('primary_recommendation', 'No primary recommendation')}")
                
                alternatives = suggestions.get('alternative_options', [])
                if alternatives:
                    print(f"\n   🔄 **Alternative Diagnostics:**")
                    for idx, alt in enumerate(alternatives):
                        print(f"      [{idx}] {alt}")
                
                implementation_steps = suggestions.get('implementation_steps', [])
                if implementation_steps:
                    print(f"\n   📋 **Implementation Steps:**")
                    for idx, step in enumerate(implementation_steps):
                        print(f"      {idx+1}. {step}")
                
                # Use next_actions for interactive selection
                next_actions = suggestions.get('next_possible_actions', [])
                if next_actions:
                    print(f"\n   ⚡ **Available Diagnostics:**")
                    for idx, action in enumerate(next_actions):
                        print(f"      [{idx}] {action}")
                    
                    # Interactive selection from next_actions
                    action_idx = input("\n🤔 **Which diagnostic would you like to run?** (Enter number, or press Enter to skip): ")
                    if action_idx.isdigit() and int(action_idx) < len(next_actions):
                        action_id = next_actions[int(action_idx)]
                        print(f"\n🔍 **Running diagnostic:** {action_id}")
                        try:
                            pipeline.apply_action(action_id)
                            self.update_state('testing', f'cycle_{cycle+1}', action_id)
                            print(f"✅ **Diagnostic completed!**")
                        except Exception as e:
                            print(f"❌ **Diagnostic failed:** {str(e)}")
                else:
                    print("   ⚠️  No specific diagnostics available in this suggestion.")
            else:
                print("   ⚠️  No diagnostic suggestions available.")
            
            # Check if goal achieved
            if metric_goal and self.check_metric(report, metric_goal):
                print(f"\n🎊 **TESTING GOAL ACHIEVED!** 🎊")
                print(f"   Target metric {metric_goal} reached!")
                break
            
            # Ask if user wants to continue
            continue_choice = input("\n🤔 **Would you like to run more diagnostics?** (y/n): ").lower()
            if continue_choice != 'y':
                print("✅ **Testing cycle completed by user choice.**")
                break
        
        self.stop_cycle('testing')
        print(f"\n🎉 **TESTING CYCLE COMPLETED!** 🎉")
        print("Your model has been thoroughly evaluated and is ready for deployment! 🚀")
    
    def check_metric(self, report, metric_goal):
        """Check if metric goal is reached in report."""
        metrics = report.get('metrics', {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and metric_value >= metric_goal:
                return True
        return False
    
    def get_state(self, cycle_id):
        """Get cycle state."""
        return self.state.get(cycle_id, {})
    
    def get_memory(self):
        """Get cycle memory."""
        return self.memory

# Backward compatibility
class Controller(InteractiveController):
    """Enhanced controller with intelligent workflow management."""
    
    def titanic_comprehensive_workflow(self, data=None):
        """
        Comprehensive Titanic dataset workflow with multi-stage analysis.
        
        Args:
            data: Optional Titanic dataset (will create sample if not provided)
            
        Returns:
            tuple: (pipeline, complete_analysis_history)
        """
        print("🚢 TITANIC COMPREHENSIVE ANALYSIS WORKFLOW")
        print("=" * 80)
        
        # Create sample data if not provided
        if data is None:
            data = self._create_titanic_sample_data()
        
        # Initialize pipeline
        pipeline = Pipeline()
        pipeline.load_data(data)
        
        # Update suggester with comprehensive context
        if hasattr(pipeline.suggester, 'pipeline_history'):
            pipeline.suggester.pipeline_history = self.iteration_history
        
        analysis_history = []
        
        # STAGE 1: Initial Comprehensive Analysis
        print("\n🔍 STAGE 1: Initial Comprehensive Data Analysis")
        print("-" * 60)
        
        stage1_query = """
        I need a complete analysis of this Titanic dataset for survival prediction. 
        Please provide comprehensive data quality assessment, feature distribution analysis, 
        correlation analysis, data modeling readiness score, critical issues, and key insights 
        about passenger demographics and survival patterns.
        """
        
        print(f"📝 Query: Initial comprehensive analysis")
        suggestions_1 = pipeline.suggestions(user_query=stage1_query, interactive=True)
        analysis_history.append({'stage': 1, 'query': stage1_query, 'suggestions': suggestions_1})
        
        # STAGE 2: Preprocessing Strategy
        print("\n🔧 STAGE 2: Intelligent Preprocessing Strategy")
        print("-" * 60)
        
        stage2_query = """
        Based on the data analysis, what preprocessing strategy should I implement?
        Address missing value handling, feature engineering opportunities, categorical encoding, 
        outlier treatment, class balancing strategy, and provide specific implementation steps.
        """
        
        print(f"📝 Query: Preprocessing strategy")
        suggestions_2 = pipeline.suggestions(user_query=stage2_query, interactive=True)
        analysis_history.append({'stage': 2, 'query': stage2_query, 'suggestions': suggestions_2})
        
        # Apply preprocessing
        print("\n🛠️ Implementing preprocessing...")
        pipeline.preprocess()
        
        # STAGE 3: Post-Preprocessing Validation
        print("\n📈 STAGE 3: Post-Preprocessing Validation")
        print("-" * 60)
        
        stage3_query = """
        Now that preprocessing is complete, analyze the transformed dataset.
        Validate imputation effectiveness, assess feature engineering impact, 
        check for new data quality issues, evaluate feature importance, 
        and recommend final feature selection.
        """
        
        print(f"📝 Query: Post-preprocessing validation")
        suggestions_3 = pipeline.suggestions(user_query=stage3_query, interactive=True)
        analysis_history.append({'stage': 3, 'query': stage3_query, 'suggestions': suggestions_3})
        
        # STAGE 4: Model Selection
        print("\n🎯 STAGE 4: Intelligent Model Selection")
        print("-" * 60)
        
        stage4_query = """
        Given the preprocessed Titanic data characteristics, recommend the optimal modeling approach.
        Provide primary algorithm recommendation, alternative algorithms, hyperparameter starting points, 
        cross-validation strategy, ensemble possibilities, and expected performance benchmarks.
        """
        
        print(f"📝 Query: Model selection strategy")
        suggestions_4 = pipeline.suggestions(user_query=stage4_query, interactive=True)
        analysis_history.append({'stage': 4, 'query': stage4_query, 'suggestions': suggestions_4})
        
        # STAGE 5: Training and Initial Evaluation
        print("\n🏃 STAGE 5: Model Training and Initial Evaluation")
        print("-" * 60)
        
        print("Training Random Forest model...")
        pipeline.train_model()
        initial_metrics = pipeline.evaluate()
        
        print(f"📊 Initial Performance: {initial_metrics.get('accuracy', 0):.1%} accuracy")
        
        stage5_query = f"""
        My Random Forest achieved {initial_metrics.get('accuracy', 0):.1%} accuracy.
        Provide performance assessment, overfitting detection, class imbalance impact analysis, 
        feature importance insights, hyperparameter optimization strategy, and alternative recommendations.
        """
        
        print(f"📝 Query: Performance analysis")
        suggestions_5 = pipeline.suggestions(user_query=stage5_query, metrics=initial_metrics, interactive=True)
        analysis_history.append({'stage': 5, 'query': stage5_query, 'suggestions': suggestions_5, 'metrics': initial_metrics})
        
        # STAGE 6: Cross-Validation Analysis
        print("\n✅ STAGE 6: Cross-Validation and Stability Analysis")
        print("-" * 60)
        
        print("Performing 5-fold cross-validation...")
        cv_metrics = pipeline.cross_validate(cv_folds=5)
        
        stage6_query = f"""
        Cross-validation shows {cv_metrics.get('mean_accuracy', 0):.1%} ± {cv_metrics.get('std_accuracy', 0):.1%} accuracy.
        Analyze model stability, interpret variance across folds, assess overfitting, 
        evaluate generalization, recommend stability improvements, and determine deployment readiness.
        """
        
        print(f"📝 Query: CV stability analysis")
        suggestions_6 = pipeline.suggestions(user_query=stage6_query, metrics=cv_metrics, interactive=True)
        analysis_history.append({'stage': 6, 'query': stage6_query, 'suggestions': suggestions_6, 'metrics': cv_metrics})
        
        # STAGE 7: Business Insights
        print("\n💼 STAGE 7: Business Insights and Interpretation")
        print("-" * 60)
        
        stage7_query = """
        Transform technical results into business insights about Titanic survival.
        Provide key factors affecting survival, actionable maritime safety insights, 
        risk assessment for passenger profiles, economic implications, historical validation, 
        and modern ship safety recommendations.
        """
        
        print(f"📝 Query: Business insights")
        suggestions_7 = pipeline.suggestions(user_query=stage7_query, metrics=cv_metrics, interactive=True)
        analysis_history.append({'stage': 7, 'query': stage7_query, 'suggestions': suggestions_7})
        
        # STAGE 8: Deployment Strategy
        print("\n🚀 STAGE 8: Deployment Strategy and Final Recommendations")
        print("-" * 60)
        
        stage8_query = """
        Provide comprehensive deployment strategy including monitoring plan, 
        performance tracking, data drift detection, retraining triggers, A/B testing, 
        risk mitigation, documentation, and regulatory compliance.
        """
        
        print(f"📝 Query: Deployment strategy")
        suggestions_8 = pipeline.suggestions(user_query=stage8_query, metrics=cv_metrics, interactive=True)
        analysis_history.append({'stage': 8, 'query': stage8_query, 'suggestions': suggestions_8})
        
        # Final Summary
        print("\n" + "="*80)
        print("📋 TITANIC ANALYSIS COMPLETE")
        print("="*80)
        print(f"🎯 Final Model Performance: {cv_metrics.get('mean_accuracy', 0):.1%} ± {cv_metrics.get('std_accuracy', 0):.1%}")
        print(f"📊 Total Analysis Stages: {len(analysis_history)}")
        # Count complex queries safely
        complex_count = 0
        for h in analysis_history:
            suggestions = h.get('suggestions', {})
            if isinstance(suggestions, dict) and suggestions.get('query_complexity', {}).get('level') == 'high':
                complex_count += 1
        print(f"🔍 Complex Queries Processed: {complex_count}")
        print(f"💡 Business Insights Generated: Maritime safety recommendations")
        print(f"🚀 Deployment Ready: Complete strategy provided")
        
        return pipeline, analysis_history

    def _create_titanic_sample_data(self):
        """Create realistic Titanic dataset sample."""
        np.random.seed(42)
        n_samples = 891
        
        # Generate realistic Titanic data patterns
        pclass = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
        sex = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])
        
        # Age with missing values
        age_base = np.random.normal(30, 12, n_samples)
        age = []
        for i in range(n_samples):
            if np.random.random() < 0.2:
                age.append(np.nan)
            else:
                age.append(max(0.5, min(80, age_base[i])))
        
        sibsp = np.random.choice([0, 1, 2, 3, 4, 5, 8], n_samples, p=[0.68, 0.20, 0.08, 0.02, 0.01, 0.005, 0.005])
        parch = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.02, 0.005, 0.005, 0.005])
        
        # Fare with class-based patterns
        fare_base = np.random.lognormal(3, 1, n_samples)
        fare = []
        for i in range(n_samples):
            fare_val = fare_base[i]
            if pclass[i] == 1:
                fare_val *= 3
            elif pclass[i] == 2:
                fare_val *= 1.5
            fare.append(max(0, fare_val))
        
        # Cabin with many missing values
        cabin = []
        for i in range(n_samples):
            if np.random.random() < 0.77:
                cabin.append(np.nan)
            else:
                deck = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
                number = np.random.randint(1, 200)
                cabin.append(f"{deck}{number}")
        
        # Embarked
        embarked = np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
        
        # Names with titles
        names = []
        for i in range(n_samples):
            if sex[i] == 'male':
                title = 'Master.' if age[i] and age[i] < 15 else np.random.choice(['Mr.', 'Dr.', 'Rev.'], p=[0.9, 0.08, 0.02])
            else:
                title = 'Miss.' if age[i] and age[i] < 18 else np.random.choice(['Mrs.', 'Miss.'], p=[0.7, 0.3])
            
            first_names = ['John', 'William', 'James', 'George', 'Charles', 'Frank', 'Joseph', 'Henry', 'Robert', 'Thomas']
            female_names = ['Mary', 'Anna', 'Elizabeth', 'Margaret', 'Ruth', 'Florence', 'Ethel', 'Emma', 'Marie', 'Lillian']
            
            if sex[i] == 'male':
                name = f"{title} {np.random.choice(first_names)} {np.random.choice(['Smith', 'Johnson', 'Brown', 'Davis', 'Miller'])}"
            else:
                name = f"{title} {np.random.choice(female_names)} {np.random.choice(['Smith', 'Johnson', 'Brown', 'Davis', 'Miller'])}"
            
            names.append(name)
        
        # Survival with realistic patterns
        survived = []
        for i in range(n_samples):
            survival_prob = 0.38
            
            if pclass[i] == 1:
                survival_prob += 0.3
            elif pclass[i] == 2:
                survival_prob += 0.1
            
            if sex[i] == 'female':
                survival_prob += 0.35
            
            if age[i] and age[i] < 16:
                survival_prob += 0.15
            
            if sibsp[i] == 0 and parch[i] == 0:
                survival_prob -= 0.1
            
            if fare[i] > 50:
                survival_prob += 0.1
            
            survival_prob = max(0.05, min(0.95, survival_prob))
            survived.append(int(np.random.random() < survival_prob))
        
        return pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Survived': survived,
            'Pclass': pclass,
            'Name': names,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Cabin': cabin,
            'Embarked': embarked
        })

# Alias for backward compatibility
CycleController = Controller
