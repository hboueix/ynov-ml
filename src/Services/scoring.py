class ScoringService:
    def __init__(self, classifier) -> None:
        self.clf = classifier
        self.last_predictions = None

    def score(self, X, y=None):
        import numpy as np
        from sklearn.metrics import mean_squared_log_error
        y_preds = self.clf.predict(X)
        self.last_predictions = y_preds
        return np.sqrt(mean_squared_log_error(y, y_preds)) if y is not None else y_preds

    def cv_score(self, X, y, cv=5):
        import numpy as np
        from sklearn.model_selection import cross_val_score
        self.last_cv_scores = np.sqrt(-1 * cross_val_score(self.clf, X, y,
                                                           cv=cv,
                                                           scoring='neg_mean_squared_log_error',
                                                           error_score=-1))
        return self.last_cv_scores.mean()
