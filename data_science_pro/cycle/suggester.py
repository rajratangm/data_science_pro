from data_science_pro.api import llm_connector
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
        if not api_key:
            raise ValueError("API key must be provided to ChainOfThoughtSuggester.")
        
        self.llm = llm_connector.LLMConnector(api_key)
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
        
        # Extract key information for context
        if isinstance(analyzer_result, dict):
            self.dataset_size = analyzer_result.get('dataset_size', 'unknown')
            self.target_type = analyzer_result.get('target_type', 'unknown')
            self.data_quality_score = analyzer_result.get('data_quality_score', 'unknown')
        
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

# Keep the original Suggester class for backward compatibility, but use the new one internally
class Suggester(ChainOfThoughtSuggester):
    """Enhanced Suggester with chain-of-thought reasoning - backward compatible."""
    pass