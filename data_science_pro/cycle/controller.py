from langgraph.graph import StateGraph, END
from data_science_pro.data.data_loader import DataLoader
from data_science_pro.data.data_analyzer import DataAnalyzer
from data_science_pro.cycle.suggester import Suggester
from data_science_pro.data.data_operations import DataOperations
from data_science_pro.modeling.trainer import Trainer
from data_science_pro.modeling.evaluator import Evaluator
from data_science_pro.cycle.reporter import Reporter
from data_science_pro.cycle.indexer import Indexer
from data_science_pro.cycle.retriever import Retriever
from data_science_pro.cycle.orchestrator import Orchestrator
from data_science_pro.cycle.planner import Planner
from data_science_pro.cycle.critic import Critic
from data_science_pro.cycle.target_selector import TargetSelector

class DataScienceController:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def run(self, csv_path: str):
        """
        Build and execute the LangGraph workflow.
        """
        graph = StateGraph(dict)

        # Register agents as graph nodes
        graph.add_node("loader", DataLoader(self.api_key))
        graph.add_node("analyzer", DataAnalyzer(self.api_key))
        graph.add_node("indexer", Indexer(self.api_key))
        graph.add_node("retriever", Retriever(self.api_key))
        graph.add_node("suggester", Suggester(self.api_key))
        graph.add_node("operations", DataOperations(self.api_key))
        graph.add_node("trainer", Trainer(self.api_key))
        graph.add_node("evaluator", Evaluator(self.api_key))
        graph.add_node("reporter", Reporter(self.api_key))
        graph.add_node("orchestrator", Orchestrator(self.api_key))
        graph.add_node("planner", Planner(self.api_key))
        graph.add_node("critic", Critic(self.api_key))
        graph.add_node("target_selector", TargetSelector(self.api_key))

        # Define workflow edges
        graph.set_entry_point("loader")
        graph.add_edge("loader", "analyzer")
        graph.add_edge("analyzer", "indexer")
        graph.add_edge("indexer", "retriever")
        graph.add_edge("retriever", "planner")
        graph.add_edge("planner", "suggester")
        graph.add_edge("suggester", "target_selector")
        graph.add_edge("target_selector", "orchestrator")
        # From here, go to orchestrator to decide next step
        # sggester now leads to target selection before orchestrator
        # Conditional routing function using next_action in state
        def route(state: dict):
            action = state.get("next_action")
            if action == "preprocess":
                return "operations"
            if action == "train":
                return "trainer"
            if action == "evaluate":
                return "evaluator"
            if action == "analyze":
                return "analyzer"
            return "reporter"

        graph.add_conditional_edges("orchestrator", route, {
            "operations": "operations",
            "trainer": "trainer",
            "evaluator": "evaluator",
            "analyzer": "analyzer",
            "reporter": "reporter",
        })

        # Linear edges among ops/train/eval loop
        graph.add_edge("operations", "trainer")
        graph.add_edge("trainer", "evaluator")
        # After evaluation, go back to orchestrator for the next decision
        graph.add_edge("evaluator", "critic")
        graph.add_edge("critic", "orchestrator")
        # Reporter ends
        graph.add_edge("reporter", END)

        # Compile and run
        app = graph.compile()
        final_report = app.invoke({
            "csv_path": csv_path,
            "iteration": 0,
            "max_iterations": 5,
            "target_metric": "accuracy",
            "target_value": 0.85,
            "user_query": getattr(self, "goal", "Automated EDA, preprocessing, training and evaluation"),
            "user_target": getattr(self, "user_target", None)
        })
        return final_report
