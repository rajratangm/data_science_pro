from data_science_pro.api.llm_connector import LLMConnector
import json

class ReportGenerator:
    
    def __init__(self, api_key):
        try:
            self.llm = LLMConnector(api_key=api_key)
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {e}")
            self.llm = None
    
    def generate_report(self, analysis_results, model_performance=None):
        """Generate AI-powered data analysis report with intelligent insights."""
        
        if self.llm is None:
            print("⚠️  LLM not available, generating basic report.")
            return self._generate_fallback_report(analysis_results, model_performance)
        
        try:
            # Create comprehensive prompt for AI analysis
            print("🧠 Generating AI-powered report...")
            prompt = self._create_analysis_prompt(analysis_results, model_performance)
            
            # Generate AI-powered insights
            ai_insights = self.llm.generate_response(prompt)
            
            # Format the final report
            report = self._format_report(ai_insights, analysis_results, model_performance)
            
            return report
            
        except Exception as e:
            print(f"⚠️  AI report generation failed: {e}")
            return self._generate_fallback_report(analysis_results, model_performance)
    
    def _create_analysis_prompt(self, analysis_results, model_performance=None):
        """Create a comprehensive prompt for AI analysis."""
        
        prompt = f"""
        You are an expert data scientist. Analyze this dataset and provide comprehensive insights:
        
        DATASET OVERVIEW:
        - Shape: {analysis_results.get('shape', 'Unknown')}
        - Columns: {len(analysis_results.get('columns', []))}
        - Numerical features: {len(analysis_results.get('feature_types', {}).get('numerical', []))}
        - Categorical features: {len(analysis_results.get('feature_types', {}).get('categorical', []))}
        
        DATA QUALITY ISSUES:
        - Missing values: {sum(analysis_results.get('missing_values', {}).values())} total
        - High missing percentage: {[col for col, pct in analysis_results.get('missing_pct', {}).items() if pct > 50]}
        - Outliers detected: {analysis_results.get('outliers', {})}
        
        KEY INSIGHTS NEEDED:
        1. What are the most important patterns in this data?
        2. What data quality issues should be addressed first?
        3. What features look most promising for modeling?
        4. Are there any obvious data integrity problems?
        5. What preprocessing steps would you recommend?
        6. What type of machine learning problem does this suggest?
        
        CORRELATION ANALYSIS:
        - Top correlations: {analysis_results.get('top_correlations', [])[:3]}
        - Any highly correlated features that might cause multicollinearity?
        
        DISTRIBUTION INSIGHTS:
        - Skewed features: {[col for col, skew in analysis_results.get('skewness', {}).items() if abs(skew) > 2]}
        - Class imbalance detected: {analysis_results.get('imbalance', False)}
        """
        
        if model_performance:
            prompt += f"""
            
            MODEL PERFORMANCE:
            {json.dumps(model_performance, indent=2)}
            
            Based on these metrics, what specific improvements would you suggest?
            """
        
        prompt += """
        
        Please provide a comprehensive, easy-to-understand analysis with:
        - Executive summary (2-3 sentences)
        - Key findings with specific numbers
        - Actionable recommendations
        - Risk assessment
        - Next steps priority list
        
        Make it engaging and professional, suitable for both technical and business audiences.
        """
        
        return prompt
    
    def _format_report(self, ai_insights, analysis_results, model_performance=None):
        """Format the final report with AI insights and raw data."""
        
        report = f"""
🧠 **AI-POWERED DATA ANALYSIS REPORT**
{'='*60}

{ai_insights}

📊 **QUICK DATA OVERVIEW:**
• Dataset shape: {analysis_results.get('shape', 'Unknown')}
• Total columns: {len(analysis_results.get('columns', []))}
• Missing values: {sum(analysis_results.get('missing_values', {}).values())} cells
• Data quality score: {self._calculate_data_quality_score(analysis_results):.1f}/10

🔍 **KEY METRICS:**
• Numerical features: {len(analysis_results.get('feature_types', {}).get('numerical', []))}
• Categorical features: {len(analysis_results.get('feature_types', {}).get('categorical', []))}
• Potential outliers: {sum(analysis_results.get('outliers', {}).values())} total

"""
        
        if model_performance:
            report += f"""
🏆 **MODEL PERFORMANCE SUMMARY:**
{json.dumps(model_performance, indent=2, default=str)}

"""
        
        report += """
💡 **NEXT STEPS:**
1. Review the AI recommendations above
2. Address critical data quality issues first
3. Consider suggested preprocessing steps
4. Evaluate feature engineering opportunities
5. Select appropriate modeling approach

🚀 **Ready to continue?** Use pipeline.suggestions() for next actions!
"""
        
        return report
    
    def _generate_fallback_report(self, analysis_results, model_performance=None):
        """Generate a basic fallback report when AI is unavailable."""
        
        report = f"""
📊 **BASIC DATA ANALYSIS REPORT**
{'='*50}

📈 **Dataset Overview:**
• Shape: {analysis_results.get('shape', 'Unknown')}
• Columns: {analysis_results.get('columns', [])}

🔢 **Data Types:**
• Numerical: {analysis_results.get('feature_types', {}).get('numerical', [])}
• Categorical: {analysis_results.get('feature_types', {}).get('categorical', [])}

⚠️ **Data Quality Issues:**
• Missing values: {analysis_results.get('missing_values', {})}
• Outliers: {analysis_results.get('outliers', {})}

📊 **Statistical Summary:**
{analysis_results.get('description', 'No description available')}

🔗 **Top Correlations:**
{analysis_results.get('top_correlations', [])[:3]}
"""
        
        if model_performance:
            report += f"\n🏆 **Model Performance:**\n{model_performance}"
        
        report += "\n\n⚠️  This is a basic report. Install langchain and openai for AI-powered insights!"
        
        return report
    
    def _calculate_data_quality_score(self, analysis_results):
        """Calculate a simple data quality score."""
        score = 10.0
        
        # Deduct points for missing values
        total_missing = sum(analysis_results.get('missing_values', {}).values())
        if total_missing > 0:
            score -= min(3.0, total_missing / 100)
        
        # Deduct points for high missing percentages
        for col, pct in analysis_results.get('missing_pct', {}).items():
            if pct > 50:
                score -= 1.0
            elif pct > 20:
                score -= 0.5
        
        # Deduct points for class imbalance
        if analysis_results.get('imbalance', False):
            score -= 1.0
        
        # Deduct points for too many outliers
        total_outliers = sum(analysis_results.get('outliers', {}).values())
        if total_outliers > 100:
            score -= 1.0
        
        return max(0.0, score)
