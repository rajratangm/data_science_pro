from ..api.llm_connector import LLMConnector
from ..data.data_loader import DataLoader
from ..data.data_analyzer import DataAnalyzer
from ..cycle.suggester import Suggester
from ..data.data_operations import DataOperations
from ..modeling.trainer import Trainer
from ..modeling.evaluator import Evaluator
from ..modeling.registry import Registry
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class DataSciencePro:
    def __init__(self, api_key=None):
        self.llm = None
        self.memory = None
        self.data = None
        self.target_col = None
        self.analyzer = DataAnalyzer()
        self.suggester = Suggester(api_key=api_key)  # Pass API key here
        self.operations = DataOperations()
        self.model_plan = None
        self.model_instance = None  # Renamed to avoid conflict with method
        self.trainer = Trainer()
        self.evaluator = Evaluator()
        self.registry = Registry()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def api(self, api_key):
        """Initializes LangChain LLM + memory."""
        self.llm = LLMConnector(api_key)
        self.memory = []

    def input_data(self, file_path, target_col):
        """Loads dataset via DataLoader."""
        loader = DataLoader()
        self.data = loader.load(file_path)
        self.target_col = target_col

    def report(self):
        """Generates dynamic analysis reports."""
        return self.analyzer.analyze(self.data)

    def suggestions(self, user_query=None, metrics=None):
        """Uses Suggester (LangChain agent) to propose next actions, with user input."""
        analyzer_result = self.report()
        if user_query is None:
            user_query = input("Enter your question or goal for this step: ")
        return self.suggester.suggest_next_action(analyzer_result, user_query, metrics)

    def apply_action(self, action_id):
        """Applies preprocessing operation using DataOperations."""
        # Use DataOperations for all preprocessing
        if action_id == 'drop_na':
            self.data = self.data.dropna()
        elif action_id == 'drop_duplicates':
            self.data = self.data.drop_duplicates()
        elif action_id == 'drop_constant':
            constant_cols = [col for col in self.data.columns if self.data[col].nunique() == 1]
            self.data = self.operations.drop_columns(self.data, constant_cols)
        elif action_id == 'drop_high_na':
            thresh = len(self.data) * 0.5
            high_na_cols = [col for col in self.data.columns if self.data[col] != self.target_col and self.data[col].isna().sum() > thresh]
            self.data = self.operations.drop_columns(self.data, high_na_cols)
        elif action_id == 'fill_na':
            for col in self.data.columns:
                if self.data[col].dtype in ['float64', 'int64']:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                else:
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        elif action_id == 'encode_categorical':
            # One-hot encode all object columns except target
            cols = [col for col in self.data.select_dtypes(include='object').columns if col != self.target_col]
            self.data = self.operations.encode(self.data, columns=cols)
        elif action_id == 'scale_numeric':
            num_cols = [col for col in self.data.select_dtypes(include='number').columns if col != self.target_col]
            self.data = self.operations.scale(self.data, columns=num_cols, method='standard')
        elif action_id == 'feature_gen':
            self.data = self.operations.generate_features(self.data)
        # After preprocessing, drop high cardinality columns and non-numeric columns
        if hasattr(self, 'target_col') and self.target_col in self.data.columns:
            high_card_cols = [col for col in self.data.columns if self.data[col].nunique() > 50 and col != self.target_col]
            self.data = self.operations.drop_columns(self.data, high_card_cols)
            non_numeric_cols = [col for col in self.data.columns if col != self.target_col and not pd.api.types.is_numeric_dtype(self.data[col])]
            self.data = self.operations.drop_columns(self.data, non_numeric_cols)
            # Train/test split
            X = self.data.drop(columns=[self.target_col])
            y = self.data[self.target_col]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def set_model(self, model_name, hyperparams):
        """Instantiate and set the model with user-selected hyperparameters."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        # Add more models as needed
        if model_name.lower() == 'randomforest':
            self.model_instance = RandomForestClassifier(**hyperparams)
        elif model_name.lower() == 'logisticregression':
            self.model_instance = LogisticRegression(**hyperparams)
        else:
            print(f"Model {model_name} not recognized. Using LogisticRegression by default.")
            self.model_instance = LogisticRegression(**hyperparams)

    def train(self):
        """Trains ML model."""
        if self.model_instance is not None and self.X_train is not None and self.y_train is not None:
            self.model_instance.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evaluates ML model."""
        if self.model_instance is not None and self.X_test is not None and self.y_test is not None:
            y_pred = self.model_instance.predict(self.X_test)
            return self.evaluator.evaluate(self.y_test, y_pred)

    def test(self):
        """Tests ML model (could be same as evaluate or extended)."""
        return self.evaluate()

    def save_model(self, path):
        """Saves trained model."""
        if self.model_instance is not None:
            self.registry.save_model(self.model_instance, 'model', 1)  # Versioning can be improved

    def run_full_cycle(self, controller, metric_goal=None):
        """
        Run full cyclic workflow: preprocessing, training, testing until metrics are satisfied.
        """
        print("Starting preprocessing cycle...")
        controller.run_preprocessing_cycle(self, metric_goal=metric_goal)
        print("Starting training cycle...")
        controller.run_training_cycle(self, metric_goal=metric_goal)
        print("Starting testing cycle...")
        controller.run_testing_cycle(self, metric_goal=metric_goal)
