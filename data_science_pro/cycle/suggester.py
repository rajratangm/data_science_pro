from data_science_pro.api.llm_connector import LLMConnector
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
import json
import pandas as pd
from typing import Dict, List, Any

class ChainOfThoughtSuggester:
    """
    Advanced AI agent that provides sophisticated, chain-of-thought reasoning for data science workflows.
    Engages users with detailed analysis and step-by-step recommendations.
    """
    
    def __init__(self, api_key=None, memory=None):
        """Initialize the suggester with optional API key."""
        if api_key:
            self.llm = LLMConnector(api_key)
        else:
            self.llm = None
        self.memory = memory or ConversationBufferMemory()
        self.conversation_history = []
        self.workflow_stage = "initial"
        self.previous_actions = []
        self.current_metrics = {}
        
        # Define comprehensive reasoning chains for different scenarios
        self.reasoning_chains = {
            "data_quality": self._data_quality_reasoning,
            "feature_engineering": self._feature_engineering_reasoning,
            "model_selection": self._model_selection_reasoning,
            "hyperparameter_tuning": self._hyperparameter_tuning_reasoning,
            "evaluation": self._evaluation_reasoning
        }
        
        self._setup_advanced_agent()
    
    def _setup_advanced_agent(self):
        """Setup sophisticated agent with specialized tools for each aspect of data science workflow."""
        
        # Only setup agent if LLM is available
        if not self.llm:
            return
            
        tools = [
            Tool(
                name="DataQualityAnalyzer",
                func=self._analyze_data_quality,
                description="Comprehensive data quality analysis with specific recommendations"
            ),
            Tool(
                name="FeatureEngineeringExpert", 
                func=self._suggest_feature_engineering,
                description="Advanced feature engineering suggestions with justification"
            ),
            Tool(
                name="ModelSelectionStrategist",
                func=self._strategic_model_selection,
                description="Strategic model selection based on data characteristics and goals"
            ),
            Tool(
                name="HyperparameterOptimizer",
                func=self._optimize_hyperparameters,
                description="Intelligent hyperparameter optimization with reasoning"
            ),
            Tool(
                name="EvaluationInterpreter",
                func=self._interpret_evaluation_results,
                description="Deep analysis of model evaluation results with next steps"
            )
        ]
        
        # Use more advanced agent type for better reasoning
        self.agent = initialize_agent(
            tools,
            OpenAI(openai_api_key=self.llm.api_key, temperature=0.1),
            agent="zero-shot-react-description",
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
    
    def suggest_next_action(self, analyzer_result: Dict, user_query: str, metrics: Dict = None) -> Dict[str, Any]:
        """
        Main method that provides comprehensive, chain-of-thought suggestions.
        
        Returns structured response with reasoning, confidence, and specific recommendations.
        """
        
        # Update conversation history and context
        self._update_context(analyzer_result, user_query, metrics)
        
        # Determine current workflow stage and appropriate reasoning chain
        current_stage = self._determine_workflow_stage()
        
        # Generate comprehensive chain-of-thought analysis
        analysis = self._generate_chain_of_thought_analysis(current_stage, analyzer_result, user_query, metrics)
        
        # Create structured, engaging response
        response = {
            "stage": current_stage,
            "reasoning": analysis["reasoning"],
            "primary_recommendation": analysis["primary_action"],
            "alternative_options": analysis["alternatives"],
            "confidence_score": analysis["confidence"],
            "detailed_steps": analysis["implementation_steps"],
            "expected_outcomes": analysis["expected_results"],
            "user_engagement": analysis["engagement_message"],
            "next_possible_actions": analysis["next_actions"]
        }
        
        # Update history
        self.previous_actions.append({
            "stage": current_stage,
            "action": analysis["primary_action"],
            "reasoning": analysis["reasoning"][:200]  # Store summary
        })
        
        return response
    
    def _generate_chain_of_thought_analysis(self, stage: str, analyzer_result: Dict, user_query: str, metrics: Dict) -> Dict[str, Any]:
        """Generate detailed chain-of-thought analysis for the current stage."""
        
        # Base analysis prompt
        base_prompt = f"""
        You are an expert data scientist with 10+ years of experience. 
        
        CURRENT CONTEXT:
        - Workflow Stage: {stage}
        - User Query: {user_query}
        - Data Analysis: {json.dumps(analyzer_result, indent=2)}
        - Current Metrics: {json.dumps(metrics, indent=2) if metrics else 'None'}
        - Previous Actions: {json.dumps(self.previous_actions[-3:], indent=2)}
        
        THINKING PROCESS:
        1. First, analyze the current data characteristics and quality issues
        2. Consider the user's specific goals and constraints  
        3. Evaluate what has been tried so far and what worked/didn't work
        4. Identify the most critical next step for maximum impact
        5. Consider alternative approaches and their trade-offs
        6. Provide specific, actionable recommendations with clear justification
        
        RESPONSE FORMAT:
        Provide your response in this exact JSON format:
        {{
            "reasoning": "Detailed explanation of your analysis and decision process",
            "primary_action": "The most important next action to take",
            "alternatives": ["Alternative approach 1", "Alternative approach 2"],
            "confidence": 0.95,
            "implementation_steps": ["Step 1", "Step 2", "Step 3"],
            "expected_results": "What improvements you expect to see",
            "engagement_message": "An engaging, encouraging message for the user",
            "next_actions": ["action1", "action2", "action3"]
        }}
        """
        
        # Get specialized analysis based on stage
        if stage in self.reasoning_chains:
            specialized_analysis = self.reasoning_chains[stage](analyzer_result, user_query, metrics)
            return specialized_analysis
        
        # Default comprehensive analysis
        response = self.llm.generate_response(base_prompt)
        
        try:
            # Parse JSON response
            parsed_response = json.loads(response)
            return parsed_response
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "reasoning": f"Based on my analysis of your data and goals, I recommend focusing on {stage}. {response}",
                "primary_action": f"optimize_{stage}",
                "alternatives": [f"alternative_{stage}_1", f"alternative_{stage}_2"],
                "confidence": 0.8,
                "implementation_steps": ["Analyze current state", "Apply recommended action", "Evaluate results"],
                "expected_results": "Improved model performance and data quality",
                "engagement_message": "Great progress! Let's optimize this further.",
                "next_actions": [f"next_{stage}_1", f"next_{stage}_2"]
            }
    
    def _data_quality_reasoning(self, analyzer_result: Dict, user_query: str, metrics: Dict) -> Dict[str, Any]:
        """Specialized reasoning for data quality issues."""
        
        # Handle string fallback case
        if isinstance(analyzer_result, str):
            print("⚠️  Warning: AI report generation failed, using basic data quality recommendations")
            return {
                "reasoning": "Data quality improvements are essential for better model performance. Focus on handling missing values, detecting outliers, and ensuring consistent data types.",
                "primary_action": "comprehensive_data_cleaning",
                "alternatives": ["missing_value_imputation", "outlier_detection", "data_type_conversion"],
                "confidence": 0.9,
                "implementation_steps": [
                    "Identify and quantify all data quality issues",
                    "Apply targeted missing value imputation",
                    "Handle outliers using domain-appropriate methods",
                    "Validate improvements with before/after comparison"
                ],
                "expected_results": "Significant improvement in model stability and performance metrics",
                "engagement_message": "🎯 I've spotted some key data quality issues that are likely holding back your model! Let's fix these systematically.",
                "next_actions": ["drop_na", "fill_na", "detect_outliers", "validate_cleaning"]
            }
        
        # If no LLM available, return default data quality suggestions
        if not self.llm:
            return {
                "reasoning": "Based on standard data science practices, focus on handling missing values, outliers, and data type consistency for better model performance.",
                "primary_action": "comprehensive_data_cleaning",
                "alternatives": ["targeted_missing_value_treatment", "outlier_detection_and_removal"],
                "confidence": 0.8,
                "implementation_steps": [
                    "Identify and quantify all data quality issues",
                    "Apply targeted missing value imputation",
                    "Handle outliers using domain-appropriate methods",
                    "Validate improvements with before/after comparison"
                ],
                "expected_results": "Significant improvement in model stability and performance metrics",
                "engagement_message": "🎯 I've spotted some key data quality issues that are likely holding back your model! Let's fix these systematically.",
                "next_actions": ["drop_na", "fill_na", "detect_outliers", "validate_cleaning"]
            }
        
        reasoning_prompt = f"""
        DATA QUALITY ANALYSIS:
        
        Current Data Issues Identified:
        - Missing Values: {analyzer_result.get('missing_values', 'Unknown')}
        - Data Types: {analyzer_result.get('data_types', 'Unknown')}
        - Cardinality Issues: {analyzer_result.get('high_cardinality', 'Unknown')}
        - Outliers: {analyzer_result.get('outliers', 'Unknown')}
        
        CHAIN OF THOUGHT:
        1. CRITICAL ISSUE IDENTIFICATION: What is the most severe data quality problem affecting model performance?
        2. IMPACT ASSESSMENT: How is each issue specifically harming your analysis?
        3. PRIORITIZATION: Which issues should be addressed first for maximum impact?
        4. METHOD SELECTION: What specific techniques will work best for YOUR data characteristics?
        5. VALIDATION STRATEGY: How will we measure improvement after fixing these issues?
        
        USER CONTEXT: {user_query}
        
        Provide detailed, actionable recommendations with specific parameter suggestions.
        """
        
        response = self.llm.generate_response(reasoning_prompt)
        
        return {
            "reasoning": f"I've identified critical data quality issues that are likely impacting your model performance. {response}",
            "primary_action": "comprehensive_data_cleaning",
            "alternatives": ["targeted_missing_value_treatment", "outlier_detection_and_removal"],
            "confidence": 0.9,
            "implementation_steps": [
                "Identify and quantify all data quality issues",
                "Apply targeted missing value imputation",
                "Handle outliers using domain-appropriate methods",
                "Validate improvements with before/after comparison"
            ],
            "expected_results": "Significant improvement in model stability and performance metrics",
            "engagement_message": "🎯 I've spotted some key data quality issues that are likely holding back your model! Let's fix these systematically.",
            "next_actions": ["drop_na", "fill_na", "detect_outliers", "validate_cleaning"]
        }
    
    def _feature_engineering_reasoning(self, analyzer_result: Dict, user_query: str, metrics: Dict) -> Dict[str, Any]:
        """Sophisticated reasoning for feature engineering opportunities."""
        
        # Handle string fallback case
        if isinstance(analyzer_result, str):
            print("⚠️  Warning: AI report generation failed, using basic feature engineering recommendations")
            return {
                "reasoning": "Feature engineering can unlock hidden patterns in your data. Consider creating interaction terms, polynomial features, and domain-specific transformations.",
                "primary_action": "strategic_feature_engineering",
                "alternatives": ["interaction_features", "polynomial_features", "domain_specific_features"],
                "confidence": 0.85,
                "implementation_steps": [
                    "Analyze current feature relationships and correlations",
                    "Create interaction terms for highly correlated features",
                    "Apply domain-specific feature transformations",
                    "Test feature selection techniques to optimize dimensionality"
                ],
                "expected_results": "Enhanced model predictive power and reduced overfitting",
                "engagement_message": "🔍 I've identified some exciting feature engineering opportunities that could unlock hidden patterns in your data!",
                "next_actions": ["feature_gen", "interaction_terms", "polynomial_features", "feature_selection"]
            }
        
        reasoning_prompt = f"""
        FEATURE ENGINEERING STRATEGY:
        
        Current Feature Analysis:
        - Numeric Features: {analyzer_result.get('numeric_features', 'Unknown')}
        - Categorical Features: {analyzer_result.get('categorical_features', 'Unknown')}
        - Feature Correlations: {analyzer_result.get('correlations', 'Unknown')}
        - Feature Importance: {analyzer_result.get('feature_importance', 'Unknown')}
        
        STRATEGIC THINKING PROCESS:
        1. DOMAIN ANALYSIS: What business/domain knowledge can inform feature creation?
        2. INTERACTION OPPORTUNITIES: Which features likely have meaningful interactions?
        3. TRANSFORMATION NEEDS: What mathematical transformations would reveal patterns?
        4. ENCODING STRATEGY: How should categorical variables be optimally encoded?
        5. DIMENSIONALITY CONSIDERATIONS: How do we balance feature richness vs. curse of dimensionality?
        
        PERFORMANCE CONTEXT: {metrics}
        USER GOALS: {user_query}
        
        Suggest specific, creative feature engineering techniques with implementation details.
        """
        
        response = self.llm.generate_response(reasoning_prompt)
        
        return {
            "reasoning": f"Your feature engineering strategy can significantly boost model performance. {response}",
            "primary_action": "strategic_feature_engineering",
            "alternatives": ["interaction_features", "polynomial_features", "domain_specific_features"],
            "confidence": 0.85,
            "implementation_steps": [
                "Analyze current feature relationships and correlations",
                "Create interaction terms for highly correlated features",
                "Apply domain-specific feature transformations",
                "Test feature selection techniques to optimize dimensionality"
            ],
            "expected_results": "Enhanced model predictive power and reduced overfitting",
            "engagement_message": "🔍 I've identified some exciting feature engineering opportunities that could unlock hidden patterns in your data!",
            "next_actions": ["feature_gen", "interaction_terms", "polynomial_features", "feature_selection"]
        }
    
    def _model_selection_reasoning(self, analyzer_result: Dict, user_query: str, metrics: Dict) -> Dict[str, Any]:
        """Intelligent model selection with detailed justification."""
        
        # Handle string fallback case
        if isinstance(analyzer_result, str):
            print("⚠️  Warning: AI report generation failed, using basic model selection recommendations")
            return {
                "reasoning": "Based on standard data science practices, Random Forest and Gradient Boosting are good starting points for most classification tasks.",
                "primary_action": "optimized_model_selection",
                "alternatives": ["ensemble_approach", "deep_learning_models", "traditional_ml"],
                "confidence": 0.88,
                "implementation_steps": [
                    "Evaluate baseline performance with simple models",
                    "Test recommended primary model with suggested hyperparameters",
                    "Compare with 2-3 alternative models for validation",
                    "Select final model based on cross-validation performance"
                ],
                "expected_results": "Optimal balance of accuracy, interpretability, and computational efficiency",
                "engagement_message": "🎯 I've analyzed your data profile and have some strategic model recommendations that should excel with your specific dataset!",
                "next_actions": ["randomforest", "logisticregression", "gradient_boosting", "neural_network"]
            }
        
        reasoning_prompt = f"""
        MODEL SELECTION INTELLIGENCE:
        
        Data Profile for Model Selection:
        - Dataset Size: {analyzer_result.get('dataset_size', 'Unknown')}
        - Feature Types: {analyzer_result.get('feature_types', 'Unknown')}
        - Target Variable: {analyzer_result.get('target_type', 'Unknown')}
        - Class Balance: {analyzer_result.get('class_balance', 'Unknown')}
        - Linearity Assessment: {analyzer_result.get('linearity', 'Unknown')}
        
        SELECTION CRITERIA ANALYSIS:
        1. DATA CHARACTERISTICS MATCH: Which models are naturally suited to your data structure?
        2. PERFORMANCE REQUIREMENTS: What accuracy/speed trade-offs meet your needs?
        3. INTERPRETABILITY NEEDS: How important is model explainability vs. raw performance?
        4. COMPUTATIONAL CONSTRAINTS: What are your time/resource limitations?
        5. ENSEMBLE OPPORTUNITIES: Would combining multiple models benefit your case?
        
        CURRENT PERFORMANCE: {metrics}
        USER REQUIREMENTS: {user_query}
        
        Recommend specific models with hyperparameter starting points and justification.
        """
        
        response = self.llm.generate_response(reasoning_prompt)
        
        return {
            "reasoning": f"Based on your data characteristics and performance requirements, here's my strategic model recommendation: {response}",
            "primary_action": "optimized_model_selection",
            "alternatives": ["ensemble_approach", "deep_learning_models", "traditional_ml"],
            "confidence": 0.88,
            "implementation_steps": [
                "Evaluate baseline performance with simple models",
                "Test recommended primary model with suggested hyperparameters",
                "Compare with 2-3 alternative models for validation",
                "Select final model based on cross-validation performance"
            ],
            "expected_results": "Optimal balance of accuracy, interpretability, and computational efficiency",
            "engagement_message": "🎯 I've analyzed your data profile and have some strategic model recommendations that should excel with your specific dataset!",
            "next_actions": ["randomforest", "logisticregression", "gradient_boosting", "neural_network"]
        }
    
    def _hyperparameter_tuning_reasoning(self, analyzer_result: Dict, user_query: str, metrics: Dict) -> Dict[str, Any]:
        """Sophisticated hyperparameter optimization reasoning."""
        
        # Handle string fallback case
        if isinstance(analyzer_result, str):
            print("⚠️  Warning: AI report generation failed, using basic hyperparameter tuning recommendations")
            return {
                "reasoning": "Hyperparameter tuning can significantly improve model performance. Start with grid search on key parameters like n_estimators, max_depth, and learning rate.",
                "primary_action": "scientific_hyperparameter_tuning",
                "alternatives": ["bayesian_optimization", "grid_search", "random_search"],
                "confidence": 0.82,
                "implementation_steps": [
                    "Identify underperforming hyperparameters from current metrics",
                    "Design targeted search space based on data characteristics",
                    "Apply efficient search strategy (Bayesian/Grid/Random)",
                    "Validate final hyperparameters with cross-validation"
                ],
                "expected_results": "Significant improvement in model generalization and performance metrics",
                "engagement_message": "⚙️ Time to fine-tune your model's engine! I've identified some hyperparameter optimizations that could give you a real performance boost.",
                "next_actions": ["hyperparameter_grid_search", "bayesian_optimization", "cross_validation"]
            }
        
        reasoning_prompt = f"""
        HYPERPARAMETER OPTIMIZATION STRATEGY:
        
        Current Model Performance: {metrics}
        Data Characteristics: {analyzer_result}
        
        OPTIMIZATION APPROACH:
        1. DIAGNOSTIC ANALYSIS: What specific model behaviors suggest hyperparameter issues?
        2. SEARCH SPACE DESIGN: Which parameters are most likely to impact your specific case?
        3. EFFICIENT SEARCH STRATEGY: Given your constraints, what's the optimal tuning approach?
        4. VALIDATION METHODOLOGY: How do we avoid overfitting during hyperparameter selection?
        5. CONVERGENCE CRITERIA: When should we stop tuning and accept current performance?
        
        Suggest specific hyperparameter ranges and tuning strategies with scientific justification.
        """
        
        response = self.llm.generate_response(reasoning_prompt)
        
        return {
            "reasoning": f"Let's optimize your model's hyperparameters systematically. {response}",
            "primary_action": "scientific_hyperparameter_tuning",
            "alternatives": ["bayesian_optimization", "grid_search", "random_search"],
            "confidence": 0.82,
            "implementation_steps": [
                "Identify underperforming hyperparameters from current metrics",
                "Design targeted search space based on data characteristics",
                "Apply efficient search strategy (Bayesian/Grid/Random)",
                "Validate final hyperparameters with cross-validation"
            ],
            "expected_results": "Significant improvement in model generalization and performance metrics",
            "engagement_message": "⚙️ Time to fine-tune your model's engine! I've identified some hyperparameter optimizations that could give you a real performance boost.",
            "next_actions": ["hyperparameter_grid_search", "bayesian_optimization", "cross_validation"]
        }
    
    def _evaluation_reasoning(self, analyzer_result: Dict, user_query: str, metrics: Dict) -> Dict[str, Any]:
        """Deep analysis of model evaluation results."""
        
        # Handle string fallback case
        if isinstance(analyzer_result, str):
            print("⚠️  Warning: AI report generation failed, using basic evaluation recommendations")
            return {
                "reasoning": "Model evaluation is crucial for understanding performance. Focus on accuracy, precision, recall, and cross-validation scores to identify improvement opportunities.",
                "primary_action": "comprehensive_model_evaluation",
                "alternatives": ["error_analysis", "cross_validation", "ensemble_methods"],
                "confidence": 0.9,
                "implementation_steps": [
                    "Deep dive into misclassification patterns",
                    "Analyze feature importance and model interpretability",
                    "Test model stability with cross-validation",
                    "Identify specific improvement opportunities"
                ],
                "expected_results": "Clear understanding of model strengths, weaknesses, and improvement opportunities",
                "engagement_message": "📊 Your model evaluation reveals some fascinating insights! Let me break down what these results mean and how we can push performance even higher.",
                "next_actions": ["detailed_error_analysis", "cross_validation", "feature_importance", "model_ensemble"]
            }
        
        reasoning_prompt = f"""
        MODEL EVALUATION DEEP DIVE:
        
        Current Evaluation Metrics: {metrics}
        Model Performance Context: {analyzer_result}
        
        EVALUATION INSIGHTS:
        1. METRIC INTERPRETATION: What do your current metrics tell us about model behavior?
        2. ERROR ANALYSIS: Where is your model making mistakes and why?
        3. GENERALIZATION ASSESSMENT: Are you overfitting or underfitting?
        4. BUSINESS IMPACT: How do these metrics translate to real-world value?
        5. IMPROVEMENT ROADMAP: What's the most impactful next step?
        
        Provide actionable insights and specific recommendations for improvement.
        """
        
        response = self.llm.generate_response(reasoning_prompt)
        
        return {
            "reasoning": f"Here's my detailed analysis of your model's performance: {response}",
            "primary_action": "comprehensive_model_evaluation",
            "alternatives": ["error_analysis", "cross_validation", "ensemble_methods"],
            "confidence": 0.9,
            "implementation_steps": [
                "Deep dive into misclassification patterns",
                "Analyze feature importance and model interpretability",
                "Test model stability with cross-validation",
                "Identify specific improvement opportunities"
            ],
            "expected_results": "Clear understanding of model strengths, weaknesses, and improvement opportunities",
            "engagement_message": "📊 Your model evaluation reveals some fascinating insights! Let me break down what these results mean and how we can push performance even higher.",
            "next_actions": ["detailed_error_analysis", "cross_validation", "feature_importance", "model_ensemble"]
        }
    
    def _update_context(self, analyzer_result: Dict, user_query: str, metrics: Dict):
        """Update internal context tracking for better personalization."""
        self.current_metrics = metrics or {}
        
        # Extract key information for context (only if analyzer_result is a dict)
        if isinstance(analyzer_result, dict):
            self.dataset_size = analyzer_result.get('dataset_size', 'unknown')
            self.target_type = analyzer_result.get('target_type', 'unknown')
            self.data_quality_score = analyzer_result.get('data_quality_score', 'unknown')
        else:
            # Set defaults when analyzer_result is a string
            self.dataset_size = 'unknown'
            self.target_type = 'unknown'
            self.data_quality_score = 'unknown'
        
        # Update workflow stage based on progress
        if not self.previous_actions:
            self.workflow_stage = "initial_analysis"
        elif any('model' in str(action) for action in self.previous_actions):
            self.workflow_stage = "model_optimization"
        elif any('feature' in str(action) for action in self.previous_actions):
            self.workflow_stage = "feature_optimization"
        else:
            self.workflow_stage = "data_preprocessing"
    
    def _determine_workflow_stage(self) -> str:
        """Determine current stage of data science workflow."""
        if not self.previous_actions:
            return "data_quality"
        elif self.current_metrics and float(self.current_metrics.get('accuracy', 0)) < 0.7:
            return "model_selection"
        elif any('hyperparameter' in str(action) for action in self.previous_actions):
            return "evaluation"
        elif any('feature' in str(action) for action in self.previous_actions):
            return "hyperparameter_tuning"
        else:
            return "feature_engineering"
    
    def _analyze_data_quality(self, data_analysis: str) -> str:
        """Analyze data quality and provide specific recommendations."""
        prompt = f"""
        Analyze this data quality assessment and provide specific recommendations:
        {data_analysis}
        
        Focus on:
        1. Most critical data quality issues
        2. Specific impact on model performance
        3. Prioritized action plan
        4. Expected improvement metrics
        """
        return self.llm.generate_response(prompt)
    
    def _suggest_feature_engineering(self, feature_analysis: str) -> str:
        """Suggest advanced feature engineering techniques."""
        prompt = f"""
        Based on this feature analysis, suggest specific feature engineering opportunities:
        {feature_analysis}
        
        Provide:
        1. Creative feature transformation ideas
        2. Interaction term suggestions
        3. Domain-specific feature recommendations
        4. Implementation details
        """
        return self.llm.generate_response(prompt)
    
    def _strategic_model_selection(self, model_context: str) -> str:
        """Provide strategic model selection advice."""
        prompt = f"""
        Given this modeling context, recommend optimal model selection strategy:
        {model_context}
        
        Include:
        1. Best model families for this data type
        2. Specific algorithm recommendations
        3. Starting hyperparameters
        4. Validation approach
        """
        return self.llm.generate_response(prompt)
    
    def _optimize_hyperparameters(self, optimization_context: str) -> str:
        """Provide hyperparameter optimization guidance."""
        prompt = f"""
        Analyze this optimization context and suggest hyperparameter tuning strategy:
        {optimization_context}
        
        Provide:
        1. Critical hyperparameters to tune
        2. Search space recommendations
        3. Efficient search strategies
        4. Stopping criteria
        """
        return self.llm.generate_response(prompt)
    
    def _interpret_evaluation_results(self, evaluation_results: str) -> str:
        """Interpret model evaluation results and suggest improvements."""
        prompt = f"""
        Analyze these evaluation results and provide actionable insights:
        {evaluation_results}
        
        Focus on:
        1. Performance interpretation
        2. Error pattern analysis
        3. Improvement opportunities
        4. Next steps prioritization
        """
        return self.llm.generate_response(prompt)

    def suggest_models(self, test_results: Dict, user_query: str, metrics: Dict = None) -> List[Dict[str, Any]]:
        """
        Suggest alternative models based on test results and performance metrics.
        
        Args:
            test_results: Current model test results and analysis
            user_query: User's improvement goals
            metrics: Current performance metrics
            
        Returns:
            List of model suggestions with reasoning
        """
        
        # If no LLM available, return default model suggestions based on metrics
        if not self.llm:
            current_accuracy = metrics.get('accuracy', 0) if metrics else 0
            
            if current_accuracy < 0.7:
                return [
                    {
                        'model': 'RandomForestClassifier',
                        'reasoning': 'Random Forest handles non-linear relationships well and is robust to overfitting. Good for improving accuracy.',
                        'expected_performance': 'Should improve accuracy by 10-20%',
                        'suggested_params': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5}
                    },
                    {
                        'model': 'GradientBoostingClassifier',
                        'reasoning': 'Gradient Boosting can capture complex patterns and often outperforms single decision trees.',
                        'expected_performance': 'Potential 15-25% accuracy improvement',
                        'suggested_params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
                    }
                ]
            else:
                return [
                    {
                        'model': 'LogisticRegression',
                        'reasoning': 'Logistic Regression is interpretable and can serve as a good baseline or ensemble component.',
                        'expected_performance': 'May provide similar performance with better interpretability',
                        'suggested_params': {'C': 1.0, 'solver': 'liblinear', 'max_iter': 1000}
                    },
                    {
                        'model': 'SVC',
                        'reasoning': 'Support Vector Classifier can work well with normalized features and might capture different patterns.',
                        'expected_performance': 'Could provide 5-10% improvement on well-separated data',
                        'suggested_params': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
                    }
                ]
        
        # LLM-powered suggestions
        prompt = f"""
        Based on these test results and metrics, suggest 3-5 alternative models:
        
        Current Test Results: {json.dumps(test_results, indent=2)}
        Current Metrics: {json.dumps(metrics, indent=2)}
        User Goals: {user_query}
        
        Consider:
        1. Current model weaknesses revealed by testing
        2. Data characteristics that might benefit from different algorithms
        3. Performance bottlenecks that alternative models could address
        4. Ensemble opportunities for better robustness
        
        Return JSON format:
        [
            {{
                "model": "ModelName",
                "reasoning": "Why this model would work better",
                "expected_performance": "Expected improvement description",
                "suggested_params": {{"param": "value"}}
            }}
        ]
        """
        
        try:
            response = self.llm.generate_response(prompt)
            suggestions = json.loads(response)
            return suggestions if isinstance(suggestions, list) else []
        except Exception:
            # Fallback to rule-based suggestions
            return self.suggest_models(test_results, user_query, metrics)  # Recursive call without LLM
    
    def suggest_hyperparams(self, test_results: Dict, model_name: str, user_query: str) -> str:
        """
        Suggest optimal hyperparameters based on test results and model performance.
        
        Args:
            test_results: Current model test results and analysis
            model_name: Name of the model to optimize
            user_query: User's improvement goals
            
        Returns:
            String representation of suggested hyperparameters dict
        """
        
        # If no LLM available, return default hyperparameters
        if not self.llm:
            default_params = {
                'RandomForestClassifier': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
                'LogisticRegression': {'C': 1.0, 'solver': 'liblinear', 'max_iter': 1000},
                'GradientBoostingClassifier': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
                'SVC': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
            }
            
            params = default_params.get(model_name, {})
            return str(params)
        
        # LLM-powered hyperparameter suggestions
        prompt = f"""
        Based on these test results, suggest optimal hyperparameters for {model_name}:
        
        Test Results: {json.dumps(test_results, indent=2)}
        User Goals: {user_query}
        
        Consider:
        1. Current performance bottlenecks revealed by testing
        2. Model-specific hyperparameter trade-offs
        3. Data characteristics affecting optimal parameter choices
        4. Computational constraints and training time considerations
        
        Return a Python dictionary string with suggested hyperparameters.
        Focus on the most impactful 3-5 parameters for this specific case.
        
        Example format: {{"n_estimators": 200, "max_depth": 15, "min_samples_split": 10}}
        """
        
        try:
            response = self.llm.generate_response(prompt)
            # Validate that response is a proper dict string
            eval(response)  # Test if it's valid Python
            return response
        except Exception:
            # Fallback to default parameters
            return self.suggest_hyperparams(test_results, model_name, user_query)  # Recursive call without LLM

# Alias for backward compatibility
Suggester = ChainOfThoughtSuggester