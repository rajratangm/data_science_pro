from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class Evaluator:
	def evaluate(self, y_true, y_pred):
		"""
		Compute common classification metrics.
		Returns a dict with accuracy, precision, recall, and F1 score.
		"""
		return {
			'accuracy': accuracy_score(y_true, y_pred),
			'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
			'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
			'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
		}

	def diagnostics(self, y_true, y_pred):
		"""
		Return confusion matrix and classification report.
		"""
		return {
			'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
			'classification_report': classification_report(y_true, y_pred, zero_division=0)
		}
