from data_science_pro.api.llm_connector import LLMConnector
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
import json
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
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
        self.pipeline_history = []  # Track complete pipeline state over time
        self.csv_analysis_cache = {}  # Cache CSV analysis results
        self.comprehensive_context = {}  # Store rich context for better suggestions
        
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
    
    def suggest_next_action(self, analyzer_result: Dict, user_query: str, metrics: Dict = None, csv_data=None) -> Dict[str, Any]:
        """
        Main method that provides comprehensive, chain-of-thought suggestions.
        
        Returns structured response with reasoning, confidence, and specific recommendations.
        """
        
        # Generate comprehensive CSV analysis if data is available
        csv_analysis = self._generate_comprehensive_csv_analysis(analyzer_result, csv_data, metrics)
        
        # Update conversation history and context with comprehensive information
        self._update_context(analyzer_result, user_query, metrics)
        
        # Determine current workflow stage and appropriate reasoning chain
        current_stage = self._determine_workflow_stage()
        
        # Generate comprehensive chain-of-thought analysis with full context
        analysis = self._generate_enhanced_chain_of_thought_analysis(current_stage, analyzer_result, user_query, metrics, csv_analysis)
        
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
            "next_possible_actions": analysis["next_actions"],
            "csv_analysis": csv_analysis,  # Include CSV insights in response
            "context_summary": self._generate_context_summary()  # Add context summary
        }
        
        # Update comprehensive history
        self._update_comprehensive_history(current_stage, analysis, metrics, csv_analysis)
        
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

    def _generate_comprehensive_csv_analysis(self, analyzer_result: Dict, csv_data: pd.DataFrame, metrics: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive CSV analysis with deep insights."""
        
        if csv_data is None or csv_data.empty:
            return {"error": "No CSV data provided for analysis"}
        
        try:
            # Basic data profiling
            analysis = {
                "dataset_overview": {
                    "shape": csv_data.shape,
                    "memory_usage": f"{csv_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                    "total_cells": csv_data.size,
                    "missing_cells": csv_data.isnull().sum().sum(),
                    "missing_percentage": (csv_data.isnull().sum().sum() / csv_data.size) * 100
                },
                
                "feature_complexity_analysis": {
                    "numeric_features": len(csv_data.select_dtypes(include=[np.number]).columns),
                    "categorical_features": len(csv_data.select_dtypes(include=['object']).columns),
                    "datetime_features": len(csv_data.select_dtypes(include=['datetime64']).columns),
                    "high_cardinality_features": self._identify_high_cardinality_features(csv_data),
                    "constant_features": self._identify_constant_features(csv_data),
                    "near_zero_variance_features": self._identify_near_zero_variance_features(csv_data)
                },
                
                "data_quality_assessment": {
                    "duplicate_rows": csv_data.duplicated().sum(),
                    "duplicate_percentage": (csv_data.duplicated().sum() / len(csv_data)) * 100,
                    "severe_missing_features": self._identify_severe_missing_features(csv_data),
                    "outlier_analysis": self._comprehensive_outlier_analysis(csv_data),
                    "data_consistency_issues": self._check_data_consistency(csv_data)
                },
                
                "statistical_insights": {
                    "feature_distributions": self._analyze_feature_distributions(csv_data),
                    "correlation_strength": self._analyze_correlation_strength(csv_data),
                    "multicollinearity_concerns": self._identify_multicollinearity(csv_data),
                    "feature_target_relationships": self._analyze_feature_target_relationships(csv_data, analyzer_result)
                },
                
                "modeling_readiness": {
                    "readiness_score": self._calculate_modeling_readiness(csv_data, metrics),
                    "critical_issues": self._identify_critical_modeling_issues(csv_data, analyzer_result),
                    "recommended_preprocessing": self._recommend_preprocessing_steps(csv_data, analyzer_result),
                    "expected_modeling_challenges": self._predict_modeling_challenges(csv_data, analyzer_result)
                }
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"CSV analysis failed: {str(e)}"}

    def _identify_high_cardinality_features(self, data: pd.DataFrame, threshold: int = 50) -> List[Dict[str, Any]]:
        """Identify features with high cardinality."""
        high_cardinality = []
        for col in data.select_dtypes(include=['object']).columns:
            unique_count = data[col].nunique()
            if unique_count > threshold:
                high_cardinality.append({
                    "feature": col,
                    "unique_values": unique_count,
                    "percentage_of_data": (unique_count / len(data)) * 100,
                    "recommendation": "Consider target encoding or frequency encoding"
                })
        return high_cardinality

    def _identify_constant_features(self, data: pd.DataFrame) -> List[str]:
        """Identify features with only one unique value."""
        constant_features = []
        for col in data.columns:
            if data[col].nunique() == 1:
                constant_features.append(col)
        return constant_features

    def _identify_near_zero_variance_features(self, data: pd.DataFrame, threshold: float = 0.01) -> List[Dict[str, Any]]:
        """Identify features with near-zero variance."""
        nzv_features = []
        for col in data.select_dtypes(include=[np.number]).columns:
            variance = data[col].var()
            if variance < threshold:
                nzv_features.append({
                    "feature": col,
                    "variance": variance,
                    "recommendation": "Consider removing or transforming"
                })
        return nzv_features

    def _identify_severe_missing_features(self, data: pd.DataFrame, threshold: float = 50.0) -> List[Dict[str, Any]]:
        """Identify features with severe missing data."""
        severe_missing = []
        missing_pct = (data.isnull().sum() / len(data)) * 100
        for col, pct in missing_pct.items():
            if pct > threshold:
                severe_missing.append({
                    "feature": col,
                    "missing_percentage": pct,
                    "missing_count": data[col].isnull().sum(),
                    "recommendation": "Consider dropping or advanced imputation"
                })
        return severe_missing

    def _comprehensive_outlier_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive outlier analysis."""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {"message": "No numeric features for outlier analysis"}
        
        outlier_analysis = {}
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (col_data < lower_bound) | (col_data > upper_bound)
                outlier_count = outliers.sum()
                
                outlier_analysis[col] = {
                    "outlier_count": int(outlier_count),
                    "outlier_percentage": (outlier_count / len(col_data)) * 100,
                    "bounds": {"lower": lower_bound, "upper": upper_bound},
                    "severity": "high" if outlier_percentage > 10 else "medium" if outlier_percentage > 5 else "low"
                }
        
        return outlier_analysis

    def _check_data_consistency(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for data consistency issues."""
        consistency_issues = []
        
        # Check for mixed data types in object columns
        for col in data.select_dtypes(include=['object']).columns:
            unique_types = set(type(x).__name__ for x in data[col].dropna())
            if len(unique_types) > 3:  # Arbitrary threshold
                consistency_issues.append({
                    "type": "mixed_data_types",
                    "feature": col,
                    "details": f"Found {len(unique_types)} different data types",
                    "recommendation": "Standardize data types or split column"
                })
        
        return consistency_issues

    def _analyze_feature_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature distributions for normality and skewness."""
        numeric_data = data.select_dtypes(include=[np.number])
        distribution_analysis = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                skewness = col_data.skew()
                kurtosis = col_data.kurtosis()
                
                distribution_analysis[col] = {
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "distribution_type": self._classify_distribution(skewness, kurtosis),
                    "transformation_suggestion": self._suggest_transformation(skewness, kurtosis)
                }
        
        return distribution_analysis

    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution based on skewness and kurtosis."""
        if abs(skewness) < 0.5:
            skew_type = "approximately_symmetric"
        elif skewness > 0.5:
            skew_type = "right_skewed"
        else:
            skew_type = "left_skewed"
        
        if abs(kurtosis) < 0.5:
            kurt_type = "normal_kurtosis"
        elif kurtosis > 0.5:
            kurt_type = "heavy_tailed"
        else:
            kurt_type = "light_tailed"
        
        return f"{skew_type}_{kurt_type}"

    def _suggest_transformation(self, skewness: float, kurtosis: float) -> str:
        """Suggest appropriate transformations based on distribution."""
        if abs(skewness) > 1.0:
            if skewness > 0:
                return "log_or_square_root"
            else:
                return "square_or_exponential"
        elif abs(kurtosis) > 2.0:
            return "robust_scaling"
        else:
            return "standard_scaling"

    def _analyze_correlation_strength(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation strength between features."""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {"message": "No numeric features for correlation analysis"}
        
        corr_matrix = numeric_data.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # Strong correlation threshold
                    strong_correlations.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_value,
                        "relationship_strength": "very_strong" if abs(corr_value) > 0.9 else "strong"
                    })
        
        return {
            "strong_correlations": strong_correlations,
            "correlation_count": len(strong_correlations),
            "recommendation": "Consider feature selection or dimensionality reduction"
        }

    def _identify_multicollinearity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify multicollinearity issues."""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {"message": "No numeric features for multicollinearity analysis"}
        
        corr_matrix = numeric_data.corr()
        
        # Find features with high correlations with multiple other features
        multicollinear_features = []
        for col in corr_matrix.columns:
            high_corr_count = (abs(corr_matrix[col]) > 0.8).sum() - 1  # Exclude self-correlation
            if high_corr_count > 2:  # Feature correlated with more than 2 others
                multicollinear_features.append({
                    "feature": col,
                    "high_correlation_count": high_corr_count,
                    "recommendation": "Consider removing or combining with correlated features"
                })
        
        return {
            "multicollinear_features": multicollinear_features,
            "severity": "high" if len(multicollinear_features) > 3 else "medium" if len(multicollinear_features) > 1 else "low"
        }

    def _analyze_feature_target_relationships(self, data: pd.DataFrame, analyzer_result: Dict) -> Dict[str, Any]:
        """Analyze relationships between features and target variable."""
        if 'target' not in analyzer_result:
            return {"message": "No target variable information available"}
        
        target_col = analyzer_result.get('target')
        if target_col not in data.columns:
            return {"message": "Target column not found in data"}
        
        relationships = {}
        
        # Analyze numeric features vs target
        numeric_features = data.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != target_col]
        
        for feature in numeric_features:
            try:
                correlation = data[feature].corr(data[target_col])
                relationships[feature] = {
                    "correlation_with_target": correlation,
                    "relationship_strength": self._classify_relationship_strength(correlation)
                }
            except:
                relationships[feature] = {"correlation_with_target": 0, "relationship_strength": "unknown"}
        
        return {
            "target_relationships": relationships,
            "strong_predictors": [f for f, rel in relationships.items() if rel.get("relationship_strength") == "strong"],
            "weak_predictors": [f for f, rel in relationships.items() if rel.get("relationship_strength") == "weak"]
        }

    def _classify_relationship_strength(self, correlation: float) -> str:
        """Classify relationship strength based on correlation."""
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            return "strong"
        elif abs_corr > 0.4:
            return "moderate"
        elif abs_corr > 0.2:
            return "weak"
        else:
            return "very_weak"

    def _calculate_modeling_readiness(self, data: pd.DataFrame, metrics: Dict = None) -> Dict[str, Any]:
        """Calculate overall modeling readiness score."""
        score = 100
        issues = []
        
        # Missing data penalty
        missing_pct = (data.isnull().sum().sum() / data.size) * 100
        if missing_pct > 20:
            score -= 30
            issues.append("High missing data percentage")
        elif missing_pct > 10:
            score -= 15
            issues.append("Moderate missing data percentage")
        
        # Duplicate data penalty
        duplicate_pct = (data.duplicated().sum() / len(data)) * 100
        if duplicate_pct > 10:
            score -= 20
            issues.append("High duplicate percentage")
        
        # High cardinality penalty
        high_cardinality = len(self._identify_high_cardinality_features(data))
        if high_cardinality > 5:
            score -= 15
            issues.append("Too many high cardinality features")
        
        # Constant features penalty
        constant_features = len(self._identify_constant_features(data))
        if constant_features > 0:
            score -= 10 * constant_features
            issues.append(f"{constant_features} constant features")
        
        # Performance bonus
        if metrics:
            accuracy = metrics.get('accuracy', 0)
            if accuracy > 0.8:
                score += 10
            elif accuracy > 0.9:
                score += 20
        
        return {
            "readiness_score": max(0, min(100, score)),
            "readiness_level": "excellent" if score >= 90 else "good" if score >= 75 else "fair" if score >= 60 else "poor",
            "issues_affecting_score": issues,
            "recommendations": self._generate_readiness_recommendations(score, issues)
        }

    def _identify_critical_modeling_issues(self, data: pd.DataFrame, analyzer_result: Dict) -> List[Dict[str, Any]]:
        """Identify critical issues that must be addressed before modeling."""
        critical_issues = []
        
        # Severe missing data
        severe_missing = self._identify_severe_missing_features(data)
        if severe_missing:
            critical_issues.extend(severe_missing)
        
        # Constant features
        constant_features = self._identify_constant_features(data)
        for feature in constant_features:
            critical_issues.append({
                "feature": feature,
                "issue": "constant_feature",
                "severity": "critical",
                "recommendation": "Remove feature - provides no information"
            })
        
        # Multicollinearity
        multicollinearity = self._identify_multicollinearity(data)
        if multicollinearity.get("severity") == "high":
            critical_issues.append({
                "issue": "severe_multicollinearity",
                "severity": "critical",
                "recommendation": "Remove highly correlated features or apply dimensionality reduction"
            })
        
        return critical_issues

    def _recommend_preprocessing_steps(self, data: pd.DataFrame, analyzer_result: Dict) -> List[Dict[str, Any]]:
        """Recommend specific preprocessing steps based on analysis."""
        preprocessing_steps = []
        
        # Missing value handling
        missing_analysis = self._identify_severe_missing_features(data)
        if missing_analysis:
            preprocessing_steps.append({
                "step": "handle_missing_values",
                "priority": "high",
                "details": f"Address missing data in {len(missing_analysis)} features",
                "methods": ["mean_imputation", "median_imputation", "forward_fill", "model_based_imputation"]
            })
        
        # Outlier handling
        outlier_analysis = self._comprehensive_outlier_analysis(data)
        if isinstance(outlier_analysis, dict) and any(info.get("outlier_percentage", 0) > 5 for info in outlier_analysis.values()):
            preprocessing_steps.append({
                "step": "handle_outliers",
                "priority": "medium",
                "details": "Address significant outlier presence",
                "methods": ["winsorization", "log_transformation", "robust_scaling"]
            })
        
        # Feature engineering
        distribution_analysis = self._analyze_feature_distributions(data)
        skewed_features = [col for col, info in distribution_analysis.items() 
                           if abs(info.get("skewness", 0)) > 1.0]
        if skewed_features:
            preprocessing_steps.append({
                "step": "feature_transformation",
                "priority": "medium",
                "details": f"Transform {len(skewed_features)} skewed features",
                "methods": ["log_transformation", "box_cox", "yeo_johnson"]
            })
        
        return preprocessing_steps

    def _predict_modeling_challenges(self, data: pd.DataFrame, analyzer_result: Dict) -> List[str]:
        """Predict potential modeling challenges."""
        challenges = []
        
        # High dimensionality
        if data.shape[1] > 50:
            challenges.append("curse_of_dimensionality")
        
        # Class imbalance
        target_col = analyzer_result.get('target')
        if target_col and target_col in data.columns:
            target_distribution = data[target_col].value_counts(normalize=True)
            if len(target_distribution) > 2 and target_distribution.min() < 0.1:
                challenges.append("severe_class_imbalance")
        
        # Multicollinearity
        multicollinearity = self._identify_multicollinearity(data)
        if multicollinearity.get("severity") != "low":
            challenges.append("multicollinearity_issues")
        
        # High missing data
        missing_pct = (data.isnull().sum().sum() / data.size) * 100
        if missing_pct > 15:
            challenges.append("missing_data_complexity")
        
        return challenges

    def _generate_readiness_recommendations(self, score: int, issues: List[str]) -> List[str]:
        """Generate recommendations based on readiness score."""
        if score >= 90:
            return ["Your data is excellent for modeling - proceed with confidence!"]
        elif score >= 75:
            return ["Good data quality - minor improvements recommended", "Focus on fine-tuning preprocessing"]
        elif score >= 60:
            return ["Fair data quality - address identified issues before modeling", "Consider feature engineering opportunities"]
        else:
            return ["Poor data quality - significant preprocessing required", "Focus on data cleaning before model development"]

    def _generate_enhanced_chain_of_thought_analysis(self, stage: str, analyzer_result: Dict, user_query: str, metrics: Dict, csv_analysis: Dict = None) -> Dict[str, Any]:
        """Generate enhanced chain-of-thought analysis with comprehensive context."""
        
        # Build comprehensive context
        context_prompt = f"""
        ENHANCED CONTEXT ANALYSIS:
        
        USER QUERY: {user_query}
        CURRENT STAGE: {stage}
        CURRENT METRICS: {json.dumps(metrics, indent=2) if metrics else 'None'}
        
        COMPREHENSIVE CSV ANALYSIS:
        {json.dumps(csv_analysis, indent=2) if csv_analysis else 'No detailed CSV analysis available'}
        
        PIPELINE HISTORY:
        {json.dumps(self.pipeline_history[-5:], indent=2) if self.pipeline_history else 'No previous pipeline history'}
        
        PREVIOUS ACTIONS:
        {json.dumps(self.previous_actions[-10:], indent=2) if self.previous_actions else 'No previous actions'}
        
        COMPREHENSIVE WORKFLOW CONTEXT:
        - Total pipeline iterations: {len(self.pipeline_history)}
        - Best metrics achieved: {self._get_best_metrics()}
        - Most successful actions: {self._get_most_successful_actions()}
        - Recurring issues: {self._identify_recurring_issues()}
        - Learning patterns: {self._identify_learning_patterns()}
        
        Based on this comprehensive context, provide chain-of-thought reasoning that considers:
        1. What has worked well in previous iterations
        2. What patterns are emerging from the data
        3. What the CSV analysis reveals about data characteristics
        4. How current metrics compare to historical performance
        5. What the user is specifically asking for
        6. What would be the most logical next step based on all evidence
        """
        
        # Use the appropriate reasoning chain based on stage
        if stage in self.reasoning_chains:
            stage_analysis = self.reasoning_chains[stage](analyzer_result, user_query, metrics)
        else:
            stage_analysis = self._default_reasoning(analyzer_result, user_query, metrics)
        
        # Enhance with comprehensive context
        enhanced_analysis = {
            "reasoning": f"{context_prompt}\n\nSTAGE-SPECIFIC ANALYSIS:\n{stage_analysis.get('reasoning', '')}",
            "primary_action": stage_analysis.get('primary_action', 'continue_workflow'),
            "alternatives": stage_analysis.get('alternatives', []),
            "confidence": stage_analysis.get('confidence', 0.8),
            "implementation_steps": stage_analysis.get('implementation_steps', []),
            "expected_results": stage_analysis.get('expected_results', 'Improved performance'),
            "engagement_message": stage_analysis.get('engagement_message', 'Continuing with enhanced workflow...'),
            "next_actions": stage_analysis.get('next_actions', [])
        }
        
        return enhanced_analysis

    def _update_comprehensive_history(self, stage: str, analysis: Dict, metrics: Dict, csv_analysis: Dict = None):
        """Update comprehensive pipeline history."""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "primary_action": analysis.get("primary_action"),
            "confidence": analysis.get("confidence"),
            "metrics": metrics.copy() if metrics else {},
            "csv_insights": csv_analysis.get("modeling_readiness", {}) if csv_analysis else {},
            "user_intent": self._extract_user_intent(),
            "success_indicators": self._calculate_success_indicators(metrics, csv_analysis)
        }
        
        self.pipeline_history.append(history_entry)
        
        # Keep only last 50 entries to prevent memory issues
        if len(self.pipeline_history) > 50:
            self.pipeline_history = self.pipeline_history[-50:]

    def _get_best_metrics(self) -> Dict[str, Any]:
        """Get the best metrics achieved so far."""
        if not self.pipeline_history:
            return {}
        
        best_metrics = {}
        for entry in self.pipeline_history:
            metrics = entry.get("metrics", {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in best_metrics or value > best_metrics[metric]:
                        best_metrics[metric] = value
        
        return best_metrics

    def _get_most_successful_actions(self) -> List[str]:
        """Identify the most successful actions based on metric improvements."""
        if len(self.pipeline_history) < 2:
            return []
        
        successful_actions = []
        for i in range(1, len(self.pipeline_history)):
            current_metrics = self.pipeline_history[i].get("metrics", {})
            previous_metrics = self.pipeline_history[i-1].get("metrics", {})
            
            if self._metrics_improved(previous_metrics, current_metrics):
                action = self.pipeline_history[i].get("primary_action")
                if action:
                    successful_actions.append(action)
        
        # Return most common successful actions
        from collections import Counter
        action_counts = Counter(successful_actions)
        return [action for action, count in action_counts.most_common(3)]

    def _identify_recurring_issues(self) -> List[str]:
        """Identify issues that keep appearing."""
        if not self.pipeline_history:
            return []
        
        recurring_issues = []
        all_issues = []
        
        for entry in self.pipeline_history:
            csv_insights = entry.get("csv_insights", {})
            critical_issues = csv_insights.get("critical_issues", [])
            all_issues.extend([issue.get("issue", "unknown") for issue in critical_issues])
        
        from collections import Counter
        issue_counts = Counter(all_issues)
        recurring_issues = [issue for issue, count in issue_counts.items() if count > 1]
        
        return recurring_issues

    def _identify_learning_patterns(self) -> Dict[str, Any]:
        """Identify learning patterns from the workflow history."""
        if len(self.pipeline_history) < 3:
            return {}
        
        patterns = {
            "improvement_trend": self._calculate_improvement_trend(),
            "stage_effectiveness": self._analyze_stage_effectiveness(),
            "action_effectiveness": self._analyze_action_effectiveness(),
            "convergence_indicators": self._check_convergence()
        }
        
        return patterns

    def _calculate_improvement_trend(self) -> str:
        """Calculate overall improvement trend."""
        if len(self.pipeline_history) < 2:
            return "insufficient_data"
        
        improvements = 0
        declines = 0
        
        for i in range(1, len(self.pipeline_history)):
            current_metrics = self.pipeline_history[i].get("metrics", {})
            previous_metrics = self.pipeline_history[i-1].get("metrics", {})
            
            if self._metrics_improved(previous_metrics, current_metrics):
                improvements += 1
            elif self._metrics_declined(previous_metrics, current_metrics):
                declines += 1
        
        if improvements > declines:
            return "improving"
        elif declines > improvements:
            return "declining"
        else:
            return "stable"

    def _analyze_stage_effectiveness(self) -> Dict[str, Any]:
        """Analyze which workflow stages are most effective."""
        stage_performance = {}
        
        for entry in self.pipeline_history:
            stage = entry.get("stage")
            metrics = entry.get("metrics", {})
            
            if stage not in stage_performance:
                stage_performance[stage] = []
            
            # Use a composite score (average of available metrics)
            metric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
            if metric_values:
                stage_performance[stage].append(np.mean(metric_values))
        
        # Calculate average performance per stage
        stage_averages = {stage: np.mean(scores) if scores else 0 
                         for stage, scores in stage_performance.items()}
        
        return stage_averages

    def _analyze_action_effectiveness(self) -> Dict[str, Any]:
        """Analyze which actions are most effective."""
        action_performance = {}
        
        for entry in self.pipeline_history:
            action = entry.get("primary_action")
            metrics = entry.get("metrics", {})
            
            if action not in action_performance:
                action_performance[action] = []
            
            metric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
            if metric_values:
                action_performance[action].append(np.mean(metric_values))
        
        # Calculate average performance per action
        action_averages = {action: np.mean(scores) if scores else 0 
                          for action, scores in action_performance.items()}
        
        return dict(sorted(action_averages.items(), key=lambda x: x[1], reverse=True)[:5])

    def _check_convergence(self) -> Dict[str, Any]:
        """Check if the workflow is converging."""
        if len(self.pipeline_history) < 5:
            return {"status": "insufficient_data"}
        
        recent_metrics = [entry.get("metrics", {}) for entry in self.pipeline_history[-5:]]
        
        # Check if metrics are stabilizing
        metric_stability = {}
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            values = [m.get(metric) for m in recent_metrics if isinstance(m.get(metric), (int, float))]
            if len(values) >= 3:
                recent_std = np.std(values[-3:])
                metric_stability[metric] = "stable" if recent_std < 0.01 else "unstable"
        
        return {
            "status": "converging" if all(s == "stable" for s in metric_stability.values()) else "exploring",
            "metric_stability": metric_stability
        }

    def _metrics_improved(self, previous: Dict, current: Dict) -> bool:
        """Check if metrics improved between iterations."""
        common_metrics = set(previous.keys()) & set(current.keys())
        improvements = 0
        
        for metric in common_metrics:
            prev_val = previous.get(metric)
            curr_val = current.get(metric)
            
            if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                if curr_val > prev_val:
                    improvements += 1
                elif curr_val < prev_val:
                    return False
        
        return improvements > 0

    def _metrics_declined(self, previous: Dict, current: Dict) -> bool:
        """Check if metrics declined between iterations."""
        common_metrics = set(previous.keys()) & set(current.keys())
        declines = 0
        
        for metric in common_metrics:
            prev_val = previous.get(metric)
            curr_val = current.get(metric)
            
            if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                if curr_val < prev_val:
                    declines += 1
                elif curr_val > prev_val:
                    return False
        
        return declines > 0

    def _extract_user_intent(self) -> str:
        """Extract user intent from recent interactions."""
        if not self.pipeline_history:
            return "unknown"
        
        # Simple heuristic: look at the most recent action and metrics
        recent_entry = self.pipeline_history[-1]
        action = recent_entry.get("primary_action", "")
        
        # Map actions to intents
        intent_mapping = {
            "comprehensive_data_cleaning": "improve_data_quality",
            "strategic_feature_engineering": "enhance_features",
            "optimized_model_selection": "improve_model",
            "scientific_hyperparameter_tuning": "optimize_performance",
            "comprehensive_model_evaluation": "evaluate_performance"
        }
        
        return intent_mapping.get(action, "general_improvement")

    def _calculate_success_indicators(self, metrics: Dict, csv_analysis: Dict) -> Dict[str, Any]:
        """Calculate indicators of success for this iteration."""
        indicators = {
            "metric_improvement": self._calculate_metric_improvement(metrics),
            "data_quality_score": csv_analysis.get("modeling_readiness", {}).get("readiness_score", 0) if csv_analysis else 0,
            "critical_issues_resolved": len(csv_analysis.get("data_quality_assessment", {}).get("severe_missing_features", [])) == 0 if csv_analysis else False,
            "confidence_boost": 0
        }
        
        # Calculate confidence boost
        if indicators["metric_improvement"] > 0:
            indicators["confidence_boost"] = min(indicators["metric_improvement"] * 10, 50)
        
        return indicators

    def _calculate_metric_improvement(self, metrics: Dict) -> float:
        """Calculate metric improvement compared to best historical performance."""
        if not metrics or not self.pipeline_history:
            return 0.0
        
        best_metrics = self._get_best_metrics()
        if not best_metrics:
            return 0.0
        
        improvements = []
        for metric, current_value in metrics.items():
            if isinstance(current_value, (int, float)) and metric in best_metrics:
                best_value = best_metrics[metric]
                if current_value > best_value:
                    improvement = (current_value - best_value) / best_value if best_value != 0 else 0
                    improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0

    def _generate_context_summary(self) -> Dict[str, Any]:
        """Generate a summary of the current comprehensive context."""
        return {
            "pipeline_iterations": len(self.pipeline_history),
            "best_performance": self._get_best_metrics(),
            "current_trend": self._calculate_improvement_trend(),
            "most_effective_actions": self._get_most_successful_actions(),
            "recurring_issues": self._identify_recurring_issues(),
            "convergence_status": self._check_convergence(),
            "overall_confidence": self._calculate_overall_confidence()
        }

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence based on historical performance."""
        if not self.pipeline_history:
            return 0.5
        
        # Consider recent performance trend
        trend = self._calculate_improvement_trend()
        base_confidence = 0.8 if trend == "improving" else 0.6 if trend == "stable" else 0.4
        
        # Consider convergence
        convergence = self._check_convergence()
        if convergence.get("status") == "converging":
            base_confidence += 0.1
        
        # Consider success rate
        if len(self.pipeline_history) > 5:
            successful_actions = len(self._get_most_successful_actions())
            success_rate = successful_actions / len(self.pipeline_history)
            base_confidence += (success_rate * 0.2)
        
        return min(1.0, base_confidence)
    
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