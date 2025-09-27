from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, log_loss, roc_auc_score

class Evaluator:
	def evaluate(self, y_true, y_pred, y_pred_proba=None):
		"""
		Compute common classification metrics.
		Returns a dict with accuracy, precision, recall, and F1 score.
		If y_pred_proba is provided, also includes probability-based metrics.
		"""
		metrics = {
			'accuracy': accuracy_score(y_true, y_pred),
			'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
			'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
			'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
		}
		
		# Add probability-based metrics if available
		if y_pred_proba is not None:
			try:
				# For binary classification
				if len(y_pred_proba.shape) == 1 or (len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2):
					if len(y_pred_proba.shape) == 2:
						proba_pos = y_pred_proba[:, 1]
					else:
						proba_pos = y_pred_proba
					metrics['log_loss'] = log_loss(y_true, proba_pos)
					metrics['auc_roc'] = roc_auc_score(y_true, proba_pos)
				# For multiclass classification
				elif len(y_pred_proba.shape) == 2:
					metrics['log_loss'] = log_loss(y_true, y_pred_proba)
			except:
				# If probability metrics fail, just skip them
				pass
		
		return metrics

	def diagnostics(self, y_true, y_pred):
		"""
		Return confusion matrix and classification report.
		"""
		return {
			'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
			'classification_report': classification_report(y_true, y_pred, zero_division=0)
		}
