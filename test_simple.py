import os
import tempfile
import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression


class TestDataSciencePro(unittest.TestCase):
    def setUp(self):
        # Create a small synthetic dataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, 'test.csv')
        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5, 6, 7, 8],
            'num2': [10, 20, 30, 40, 50, 60, 70, 80],
            'cat': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_public_imports(self):
        # Package-level imports
        import data_science_pro
        from data_science_pro import DataSciencePro, run_pipeline
        from data_science_pro.data import DataAnalyzer, DataLoader, DataOperations
        from data_science_pro.modeling import Trainer, Evaluator, ModelRegistry
        from data_science_pro.cycle import DataScienceController, Suggester, ChainOfThoughtSuggester, Reporter

        self.assertIsNotNone(data_science_pro)
        self.assertTrue(callable(DataSciencePro))
        self.assertTrue(callable(run_pipeline))
        self.assertTrue(callable(DataAnalyzer))
        self.assertTrue(callable(DataLoader))
        self.assertTrue(callable(DataOperations))
        self.assertTrue(callable(Trainer))
        self.assertTrue(callable(Evaluator))
        self.assertTrue(callable(ModelRegistry))
        self.assertTrue(callable(DataScienceController))
        self.assertTrue(callable(Suggester))
        self.assertTrue(callable(ChainOfThoughtSuggester))
        self.assertTrue(callable(Reporter))

    def test_analyzer_outputs(self):
        from data_science_pro.data.data_loader import DataLoader
        from data_science_pro.data.data_analyzer import DataAnalyzer

        state = {'csv_path': self.csv_path}
        loader = DataLoader(api_key='test')
        analyzer = DataAnalyzer(api_key='test')
        state = loader(state)
        state = analyzer(state)

        self.assertIn('analysis', state)
        analysis = state['analysis']
        self.assertIn('missing_values', analysis)
        self.assertIn('unique_counts', analysis)
        self.assertIn('numeric_columns', analysis)
        self.assertIn('categorical_columns', analysis)
        self.assertIn('target_candidates', analysis)

    def test_preprocessing_preserves_target(self):
        from data_science_pro.data.data_loader import DataLoader
        from data_science_pro.data.data_analyzer import DataAnalyzer
        from data_science_pro.data.data_operations import DataOperations

        state = {'csv_path': self.csv_path}
        state = DataLoader('test')(state)
        state = DataAnalyzer('test')(state)
        state = DataOperations('test')(state)

        self.assertIn('data', state)
        self.assertIn('target', state)
        self.assertIn(state['target'], state['data'].columns)

    def test_trainer_and_evaluator(self):
        from data_science_pro.data.data_loader import DataLoader
        from data_science_pro.data.data_analyzer import DataAnalyzer
        from data_science_pro.data.data_operations import DataOperations
        from data_science_pro.modeling.trainer import Trainer
        from data_science_pro.modeling.evaluator import Evaluator

        state = {'csv_path': self.csv_path}
        state = DataLoader('test')(state)
        state = DataAnalyzer('test')(state)
        state = DataOperations('test')(state)
        state = Trainer('test')(state)
        state = Evaluator('test')(state)

        self.assertIn('evaluation', state)
        self.assertIn('accuracy', state['evaluation'])
        self.assertGreaterEqual(state['evaluation']['accuracy'], 0.0)

    def test_model_registry(self):
        from data_science_pro.modeling.registry import ModelRegistry

        reg = ModelRegistry(registry_dir=os.path.join(self.temp_dir.name, 'registry'))
        model = LogisticRegression()
        path = reg.save_model(model, name='dummy', version=1)
        self.assertTrue(os.path.exists(path))
        loaded = reg.load_model('dummy', 1)
        self.assertIsInstance(loaded, LogisticRegression)

    def test_controller_end_to_end_with_mocked_llm(self):
        # Monkeypatch LLMConnector.run to deterministic behavior driving the loop
        from data_science_pro.api.llm_connector import LLMConnector
        from data_science_pro.cycle.controller import DataScienceController

        decisions = ['preprocess', 'train', 'evaluate', 'report']

        original_run = LLMConnector.run

        def fake_run(self, prompt: str, context: dict = None, retries: int = 2, backoff_sec: float = 1.0):
            # Orchestrator: return sequential decisions
            if 'Available actions' in prompt or 'Return format' in prompt:
                return decisions.pop(0) if decisions else 'report'
            # Reporter: return a mock report
            if 'Produce a professional report in Markdown' in prompt or 'Senior Data Science Report Writer' in prompt:
                return "# Report\nFinal accuracy: 0.80\nEverything looks good."
            # Critic/Planner/Suggester: return non-blocking text
            return "OK"

        try:
            LLMConnector.run = fake_run
            controller = DataScienceController(api_key='test')
            # Ensure goal is passed
            controller.goal = "Reach accuracy >= 0.8"
            result = controller.run(self.csv_path)
            # Expect final state dict including report
            self.assertIsInstance(result, dict)
            self.assertIn('report', result)
            self.assertTrue(str(result['report']).startswith('#'))
        finally:
            LLMConnector.run = original_run


if __name__ == '__main__':
    unittest.main()
