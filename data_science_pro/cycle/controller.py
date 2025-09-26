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
            'timestamp': datetime.now()
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
            if suggestions:
                for idx, suggestion in enumerate(suggestions):
                    print(f"\n   [{idx}] **{suggestion.get('title', 'Action')}**")
                    print(f"       📝 {suggestion.get('description', 'No description')}")
                    print(f"       ⚡ Action: {suggestion.get('action_id', 'unknown')}")
                    if 'expected_impact' in suggestion:
                        print(f"       📈 Expected Impact: {suggestion['expected_impact']}")
            else:
                print("   ⚠️  No specific suggestions available. Try general goals like 'improve data quality'.")
            
            # Interactive selection
            action_idx = input("\n🤔 **Which action would you like to apply?** (Enter number, or press Enter to skip): ")
            if action_idx.isdigit() and int(action_idx) < len(suggestions):
                action_id = suggestions[int(action_idx)].get('action_id', None)
                if action_id:
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
            if suggestions:
                for idx, suggestion in enumerate(suggestions):
                    print(f"\n   [{idx}] **{suggestion.get('model', 'Model')}**")
                    print(f"       📝 {suggestion.get('reasoning', 'No reasoning provided')}")
                    if 'expected_performance' in suggestion:
                        print(f"       📈 Expected Performance: {suggestion['expected_performance']}")
            
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
            if suggestions:
                for idx, suggestion in enumerate(suggestions):
                    print(f"\n   [{idx}] **{suggestion.get('title', 'Analysis')}**")
                    print(f"       📝 {suggestion.get('description', 'No description')}")
                    print(f"       ⚡ Action: {suggestion.get('action_id', 'unknown')}")
            
            # Apply diagnostic actions
            action_idx = input("\n🤔 **Which diagnostic would you like to run?** (Enter number, or press Enter to skip): ")
            if action_idx.isdigit() and int(action_idx) < len(suggestions):
                action_id = suggestions[int(action_idx)].get('action_id', None)
                if action_id:
                    print(f"\n🔍 **Running diagnostic:** {action_id}")
                    try:
                        pipeline.apply_action(action_id)
                        self.update_state('testing', f'cycle_{cycle+1}', action_id)
                        print(f"✅ **Diagnostic completed!**")
                    except Exception as e:
                        print(f"❌ **Diagnostic failed:** {str(e)}")
            
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
    pass
