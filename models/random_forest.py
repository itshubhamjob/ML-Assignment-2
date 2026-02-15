from models.decision_tree import CustomDecisionTree
import numpy as np

class CustomRandomForest:
    def __init__(self, num_decision_trees=10, max_tree_depth=10, min_leaf_samples=2):
        self.num_decision_trees = num_decision_trees
        self.max_tree_depth = max_tree_depth
        self.min_leaf_samples = min_leaf_samples
        self.ensemble_trees = []

    def fit(self, training_features, training_labels):
        for tree_iteration in range(self.num_decision_trees):
            tree_model = CustomDecisionTree(max_tree_depth=self.max_tree_depth, min_leaf_samples=self.min_leaf_samples)
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(training_features), len(training_features), replace=True)
            tree_model.fit(training_features[bootstrap_indices], training_labels[bootstrap_indices])
            self.ensemble_trees.append(tree_model)

    def predict(self, test_features):
        ensemble_predictions = np.array([tree_model.predict(test_features) for tree_model in self.ensemble_trees])
        # Majority vote
        return np.round(np.mean(ensemble_predictions, axis=0)).astype(int)

def run_random_forest(feature_matrix, target_vector):
    model_instance = CustomRandomForest(num_decision_trees=5)
    model_instance.fit(feature_matrix.values, target_vector.values)
    ensemble_predictions = model_instance.predict(feature_matrix.values)
    return ensemble_predictions, ensemble_predictions