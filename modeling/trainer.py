from sklearn.base import BaseEstimator

class Trainer:
    def train(self, model: BaseEstimator, X_train, y_train):
        """
        Train a scikit-learn model with provided data.
        Args:
            model: scikit-learn estimator
            X_train: training features
            y_train: training labels
        Returns:
            Trained model
        """
        model.fit(X_train, y_train)
        return model
