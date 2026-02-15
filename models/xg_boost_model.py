import numpy as np
from models.decision_tree import CustomDecisionTree

class CustomGradientBoost:
    def __init__(self, num_boosting_rounds=5, boosting_shrinkage_rate=0.1, weak_learner_depth=3):
        self.num_boosting_rounds = num_boosting_rounds
        self.boosting_shrinkage_rate = boosting_shrinkage_rate
        self.weak_learner_depth = weak_learner_depth
        self.sequential_weak_learners = []
        self.initial_prediction = None

    def fit(self, training_features, training_labels):
        self.initial_prediction = np.mean(training_labels)
        ensemble_predictions = np.full(len(training_labels), self.initial_prediction)

        for boosting_iteration in range(self.num_boosting_rounds):
            # Calculate residuals
            prediction_residuals = training_labels - ensemble_predictions
            weak_learner = CustomDecisionTree(max_tree_depth=self.weak_learner_depth)
            weak_learner.fit(training_features, prediction_residuals)
            self.sequential_weak_learners.append(weak_learner)
            ensemble_predictions += self.boosting_shrinkage_rate * weak_learner.predict(training_features)

    def predict(self, test_features):
        ensemble_predictions = np.full(test_features.shape[0], self.initial_prediction)
        for weak_learner in self.sequential_weak_learners:
            ensemble_predictions += self.boosting_shrinkage_rate * weak_learner.predict(test_features)
        return [1 if prediction_value > 0.5 else 0 for prediction_value in ensemble_predictions], ensemble_predictions

def run_xgboost(feature_matrix, target_vector):
    model_instance = CustomGradientBoost()
    model_instance.fit(feature_matrix.values, target_vector.values)
    return model_instance.predict(feature_matrix.values)