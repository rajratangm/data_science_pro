def comprehensive_eda_prompt(data_summary, target_info, quality_issues, user_goals):
    """Generate comprehensive EDA analysis with engaging insights."""
    return f"""
    🎯 **EXPERT DATA ANALYSIS - Chain of Thought Reasoning**
    
    **YOUR DATA STORY:**
    - Dataset Overview: {data_summary}
    - Target Variable Analysis: {target_info}
    - Data Quality Assessment: {quality_issues}
    - User Goals: {user_goals}
    
    **SYSTEMATIC ANALYSIS PROCESS:**
    
    **Step 1: Data Characteristics Understanding**
    - What type of problem are we solving? (Classification/Regression)
    - What's the data volume and feature complexity?
    - Any domain-specific considerations I should know?
    
    **Step 2: Quality Issues Impact Assessment**
    - How severe are the missing values and what's their pattern?
    - Are there any data integrity red flags?
    - What's the class distribution and balance situation?
    
    **Step 3: Feature Relationship Discovery**
    - Which features show the strongest relationships with target?
    - Any surprising correlations or interactions?
    - What feature engineering opportunities exist?
    
    **Step 4: Strategic Recommendations**
    - What's the most critical issue to address first?
    - Which preprocessing steps will have maximum impact?
    - What modeling approach suits this data profile?
    
    **ENGAGING INSIGHTS:**
    Provide your analysis in a conversational, encouraging tone that makes the user excited about their data discoveries. Use analogies, examples, and clear explanations.
    
    **DELIVERABLES:**
    1. **Key Findings** (3-5 bullet points with emojis)
    2. **Priority Action Items** (ranked by impact)
    3. **Expected Outcomes** (what improvements to expect)
    4. **Next Steps** (specific actions with reasoning)
    
    Make this analysis feel like a collaborative data science session! 🚀
    """

def feature_suggestion_prompt(analyzer_result, user_query, metrics, history=None):
    return f"""
    🔍 **FEATURE ENGINEERING INTELLIGENCE - Deep Dive Analysis**
    
    **CURRENT SITUATION:**
    Dataset Analysis: {analyzer_result}
    User Query: {user_query}
    Target Metrics: {metrics}
    Previous Actions: {history if history else 'None'}
    
    **CHAIN OF THOUGHT REASONING:**
    
    **Phase 1: Feature Archaeology** 🏛️
    - What hidden gems are buried in your current features?
    - Which features have untapped potential?
    - What domain knowledge can we encode?
    
    **Phase 2: Relationship Mapping** 🗺️
    - Which feature combinations might reveal new patterns?
    - What interactions could be meaningful in your domain?
    - How can we capture non-linear relationships?
    
    **Phase 3: Feature Alchemy** ⚗️
    - What mathematical transformations make sense?
    - How can we handle categorical variables creatively?
    - What temporal or spatial features exist?
    
    **Phase 4: Quality vs. Quantity** ⚖️
    - How do we balance feature richness vs. overfitting?
    - What's the optimal feature selection strategy?
    - Which features should we engineer vs. eliminate?
    
    **ENGAGING DELIVERABLES:**
    1. **🎯 Top 3 Feature Opportunities** (with specific implementation)
    2. **💡 Creative Engineering Ideas** (domain-specific suggestions)
    3. **📊 Expected Impact** (quantified improvements)
    4. **⚡ Implementation Roadmap** (step-by-step guide)
    
    Think like a detective solving a mystery - what clues are hidden in your data? 🕵️‍♂️
    """

def model_suggestion_prompt(analyzer_result, user_query, metrics, history=None):
    return f"""
    🧠 **STRATEGIC MODEL SELECTION - AI-Powered Decision Making**
    
    **DECISION CONTEXT:**
    Dataset Analysis: {analyzer_result}
    User Query: {user_query}
    Target Metrics: {metrics}
    Previous Actions: {history if history else 'None'}
    
    **SOPHISTICATED REASONING CHAIN:**
    
    **Step 1: Data Profile Analysis** 📊
    - Dataset size and complexity assessment
    - Feature type distribution (numeric vs. categorical)
    - Target variable characteristics
    - Class balance and distribution patterns
    
    **Step 2: Problem Type Classification** 🎯
    - Classification vs. regression determination
    - Multi-class vs. binary considerations
    - Time series vs. cross-sectional implications
    - Online vs. batch prediction requirements
    
    **Step 3: Performance Requirements Analysis** ⚡
    - Accuracy vs. interpretability trade-offs
    - Training time constraints
    - Prediction speed requirements
    - Memory and computational limitations
    
    **Step 4: Model Family Evaluation** 🏆
    - Linear models: When simplicity wins
    - Tree-based models: For non-linear patterns
    - Neural networks: For complex relationships
    - Ensemble methods: For robust performance
    
    **Step 5: Hyperparameter Strategy** 🔧
    - Critical parameters for each model type
    - Efficient search space design
    - Cross-validation approach
    - Early stopping criteria
    
    **ENGAGING RECOMMENDATIONS:**
    1. **🏆 Primary Model Choice** (with detailed justification)
    2. **🥈 Backup Options** (2-3 alternatives with trade-offs)
    3. **📈 Expected Performance** (realistic benchmarks)
    4. **🚀 Implementation Strategy** (specific hyperparameters)
    
    Think of this as assembling your data science dream team! 👥
    """

def oversample_prompt(analyzer_result, user_query, metrics, history=None):
    return f"""
    ⚖️ **CLASS BALANCE OPTIMIZATION - Strategic Data Balancing**
    
    **BALANCING CHALLENGE:**
    Dataset Analysis: {analyzer_result}
    User Query: {user_query}
    Target Metrics: {metrics}
    Previous Actions: {history if history else 'None'}
    
    **COMPREHENSIVE REASONING PROCESS:**
    
    **Phase 1: Imbalance Severity Assessment** 📏
    - Exact class distribution percentages
    - Impact on model bias toward majority class
    - Business cost of false positives vs. false negatives
    - Sample size adequacy for minority class learning
    
    **Phase 2: Technique Selection Strategy** 🎯
    - When oversampling vs. undersampling makes sense
    - SMOTE vs. ADASYN vs. Random oversampling trade-offs
    - Ensemble balancing approaches (BalancedRandomForest)
    - Cost-sensitive learning alternatives
    
    **Phase 3: Implementation Planning** 🛠️
    - Optimal balancing ratios for your specific case
    - Cross-validation strategy for imbalanced data
    - Evaluation metrics that matter (Precision, Recall, F1, AUC)
    - Validation set balancing considerations
    
    **Phase 4: Risk Assessment** ⚠️
    - Overfitting risks with synthetic samples
    - Information leakage prevention
    - Generalization impact assessment
    - Business logic validation
    
    **STRATEGIC RECOMMENDATIONS:**
    1. **🎯 Primary Balancing Strategy** (specific technique + parameters)
    2. **📊 Expected Impact** (improvement in key metrics)
    3. **⚠️ Risk Mitigation** (how to avoid common pitfalls)
    4. **✅ Validation Approach** (how to measure success)
    
    Let's achieve the perfect balance for optimal model performance! 🎪
    """

def cleaning_process_prompt(data_state, quality_metrics, user_goals, domain_context=None):
    """Generate engaging cleaning process guidance."""
    return f"""
    🧹 **DATA CLEANING MASTERCLASS - Transforming Raw Data to Gold**
    
    **CLEANING CHALLENGE:**
    Current Data State: {data_state}
    Quality Metrics: {quality_metrics}
    User Goals: {user_goals}
    Domain Context: {domain_context if domain_context else 'General'}
    
    **SYSTEMATIC CLEANING STRATEGY:**
    
    **🕵️‍♂️ Detective Work: Issue Identification**
    - Missing Value Patterns: Random vs. Systematic vs. Structural
    - Outlier Investigation: Genuine vs. Error vs. Natural Variation
    - Inconsistency Detection: Format variations, unit mismatches
    - Duplicate Analysis: Exact vs. Fuzzy duplicates
    
    **⚗️ Cleaning Chemistry: Choosing the Right Methods**
    - Missing Values: When to impute vs. delete vs. flag
    - Outlier Treatment: Robust scaling vs. removal vs. transformation
    - Categorical Cleaning: Standardization vs. encoding strategies
    - Text Processing: Normalization vs. standardization approaches
    
    **🎯 Impact-Focused Prioritization**
    - Which issues most affect your target variable?
    - What cleaning will maximize model performance?
    - How to preserve important signal while removing noise?
    - Business logic validation throughout the process
    
    **📈 Success Measurement**
    - Before/after data quality scores
    - Model performance improvement tracking
    - Data integrity validation checks
    - Business rule compliance verification
    
    **ENGAGING ACTION PLAN:**
    1. **🔍 Issue Discovery Report** (what I found in your data)
    2. **🎯 Priority Cleaning Steps** (ranked by impact)
    3. **⚡ Quick Wins** (high-impact, low-effort actions)
    4. **🚀 Advanced Techniques** (sophisticated cleaning methods)
    
    Let's turn your data mess into data success! ✨
    """

def training_process_prompt(model_state, performance_metrics, resource_constraints, user_objectives):
    """Generate engaging training process optimization guidance."""
    return f"""
    🏋️‍♂️ **MODEL TRAINING OPTIMIZATION - Building Your AI Champion**
    
    **TRAINING CONTEXT:**
    Model State: {model_state}
    Performance Metrics: {performance_metrics}
    Resource Constraints: {resource_constraints}
    User Objectives: {user_objectives}
    
    **COMPREHENSIVE TRAINING STRATEGY:**
    
    **🧠 Training Philosophy: Art + Science**
    - Bias-Variance Balance: Understanding the fundamental trade-off
    - Learning Curve Analysis: When to stop training for optimal generalization
    - Validation Strategy: How to avoid overfitting while maximizing performance
    - Hyperparameter Philosophy: Which parameters matter most for your case?
    
    **⚡ Optimization Techniques**
    - Learning Rate Scheduling: Finding the perfect pace
    - Regularization Strategies: L1, L2, Dropout, Early Stopping
    - Ensemble Methods: Combining models for robust performance
    - Cross-Validation Approaches: Maximizing training data usage
    
    **🎯 Performance Maximization**
    - Feature Importance Leverage: Using insights for better training
    - Class Balance Handling: Ensuring fair learning across classes
    - Computational Efficiency: Training faster without sacrificing quality
    - Convergence Monitoring: Knowing when training is truly complete
    
    **📊 Progress Tracking & Validation**
    - Training/Validation Curves: Interpreting learning patterns
    - Metric Selection Strategy: Choosing the right success measures
    - Error Analysis: Learning from model mistakes
    - Business Metric Alignment: Ensuring model success translates to value
    
    **TRAINING MASTERY PLAN:**
    1. **🎯 Training Strategy Design** (customized to your data)
    2. **⚡ Optimization Roadmap** (step-by-step improvements)
    3. **📈 Performance Monitoring** (how to track progress)
    4. **🏆 Success Criteria** (when to declare victory)
    
    Time to train your model like a pro athlete! 🏆
    """
