class Controller:
    """
    Orchestrates cycles, manages memory and states.
    """
    def __init__(self):
        self.state = {}
        self.memory = []
        self.current_cycle = None

    def start_cycle(self, cycle_id):
        self.current_cycle = cycle_id
        self.state[cycle_id] = {'status': 'started'}
        self.memory.append(f"Cycle {cycle_id} started.")

    def stop_cycle(self, cycle_id):
        if cycle_id in self.state:
            self.state[cycle_id]['status'] = 'stopped'
            self.memory.append(f"Cycle {cycle_id} stopped.")
            if self.current_cycle == cycle_id:
                self.current_cycle = None

    def update_state(self, cycle_id, key, value):
        if cycle_id not in self.state:
            self.state[cycle_id] = {}
        self.state[cycle_id][key] = value
        self.memory.append(f"State updated for {cycle_id}: {key}={value}")

    def get_state(self, cycle_id):
        return self.state.get(cycle_id, {})

    def get_memory(self):
        return self.memory

    def run_preprocessing_cycle(self, pipeline, max_loops=5, metric_goal=None):
        """
        Cyclic preprocessing: report → suggestions → user action → report → repeat until satisfied or max_loops.
        """
        for cycle in range(max_loops):
            report = pipeline.report()
            print(f"Cycle {cycle+1} Report:", report)
            user_query = input("What would you like to achieve in preprocessing? (e.g., drop columns, scale, encode): ")
            suggestions = pipeline.suggestions(user_query=user_query)
            print(f"Suggestions:", suggestions)
            # User selects action interactively
            if suggestions:
                print("Available actions:")
                for idx, suggestion in enumerate(suggestions):
                    print(f"[{idx}] {suggestion}")
                action_idx = input("Select action index to apply (or press Enter to skip): ")
                if action_idx.isdigit() and int(action_idx) < len(suggestions):
                    action_id = suggestions[int(action_idx)].get('action_id', None)
                    if action_id:
                        pipeline.apply_action(action_id)
                        self.update_state('preprocessing', f'cycle_{cycle+1}', action_id)
            if metric_goal and self.check_metric(report, metric_goal):
                print(f"Metric goal {metric_goal} reached. Stopping preprocessing cycle.")
                break

    def run_training_cycle(self, pipeline, max_loops=5, metric_goal=None):
        """
        Cyclic training: train → report → suggestions → model selection → hyperparameter suggestion/input → train until satisfied or max_loops.
        """
        for cycle in range(max_loops):
            pipeline.train()
            report = pipeline.report()
            print(f"Training Cycle {cycle+1} Report:", report)
            user_query = input("What is your training goal or question? (e.g., improve accuracy, try new model): ")
            suggestions = pipeline.suggester.suggest_models(report, user_query, metrics=None)
            print(f"Model Suggestions:", suggestions)
            model_name = input("Enter the model name to implement (e.g., RandomForest, LogisticRegression): ")
            # Suggest hyperparameters using LLM
            hyperparam_suggestion = pipeline.suggester.suggest_hyperparams(report, model_name, user_query)
            print(f"Suggested Hyperparameters: {hyperparam_suggestion}")
            hyperparams = input("Enter hyperparameters as a Python dict (or press Enter to use suggestion): ")
            try:
                hyperparams_dict = eval(hyperparams) if hyperparams else eval(hyperparam_suggestion)
            except Exception:
                hyperparams_dict = {}
            pipeline.set_model(model_name, hyperparams_dict)
            pipeline.train()
            print("Model trained with your selection.")
            if metric_goal and self.check_metric(report, metric_goal):
                print(f"Metric goal {metric_goal} reached. Stopping training cycle.")
                break

    def run_testing_cycle(self, pipeline, max_loops=5, metric_goal=None):
        """
        Cyclic testing: test → report → suggestions → fix → re-test until satisfied or max_loops.
        """
        for cycle in range(max_loops):
            pipeline.test()
            report = pipeline.report()
            print(f"Testing Cycle {cycle+1} Report:", report)
            user_query = input("What is your testing goal or question? (e.g., check precision, test new features): ")
            suggestions = pipeline.suggestions(user_query=user_query)
            print(f"Suggestions:", suggestions)
            if suggestions:
                print("Available actions:")
                for idx, suggestion in enumerate(suggestions):
                    print(f"[{idx}] {suggestion}")
                action_idx = input("Select action index to apply (or press Enter to skip): ")
                if action_idx.isdigit() and int(action_idx) < len(suggestions):
                    action_id = suggestions[int(action_idx)].get('action_id', None)
                    if action_id:
                        pipeline.apply_action(action_id)
                        self.update_state('testing', f'cycle_{cycle+1}', action_id)
            if metric_goal and self.check_metric(report, metric_goal):
                print(f"Metric goal {metric_goal} reached. Stopping testing cycle.")
                break

    def check_metric(self, report, metric_goal):
        """Check if metric goal is reached in report (customize for your metric structure)."""
        # Example: check accuracy in report['metrics']
        metrics = report.get('metrics', {})
        accuracy = metrics.get('accuracy', None)
        return accuracy is not None and accuracy >= metric_goal
