from api import llm_connector
class ReportGenerator:
    
    def generate_report(self, analysis_results, model_performance= False):
        if model_performance:
            report = f"Data Analysis Report:\n{analysis_results}\nModel Performance Metrics:\n{model_performance}"
            report = llm_connector.LLMConnector().generate_response(report)
        else:
            report = f"Data Analysis Report:\n{analysis_results}"
            report = llm_connector.LLMConnector().generate_response(report)
        return report
