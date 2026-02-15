import numpy as np
from collections import Counter

class Node:
    __slots__ = ['feature', 'threshold', 'left', 'right', 'value']
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    

class CustomDecisionTree:
    def __init__(self, max_tree_depth=5, min_leaf_samples=10, num_quantile_bins=20):
        self.max_tree_depth = max_tree_depth
        self.min_leaf_samples = min_leaf_samples
        self.root_node = None
        self.num_quantile_bins = num_quantile_bins

    def fit(self, feature_data, target_data):
        target_data = target_data.astype(np.float64)
        self.root_node = self._grow_tree(feature_data, target_data)

    def _grow_tree(self, feature_data, target_data, depth_level=0):
        num_samples, num_features = feature_data.shape
        unique_labels = len(np.unique(target_data))

        if (depth_level >= self.max_tree_depth or num_samples < self.min_leaf_samples or np.var(target_data) < 1e-7):
            leaf_prediction = np.mean(target_data)
            return Node(value=leaf_prediction)

        feature_indices = np.random.choice(num_features, num_features, replace=False)
        optimal_feature, optimal_threshold = self._best_split(feature_data, target_data, feature_indices)

        left_indices, right_indices = self._split(feature_data[:, optimal_feature], optimal_threshold)
        left_subtree = self._grow_tree(feature_data[left_indices, :], target_data[left_indices], depth_level + 1)
        right_subtree = self._grow_tree(feature_data[right_indices, :], target_data[right_indices], depth_level + 1)
        return Node(optimal_feature, optimal_threshold, left_subtree, right_subtree)

    def _best_split(self, feature_data, target_data, feature_indices):
        best_information_gain = -1
        best_feature_idx, best_split_threshold = None, None
        for feature_column_idx in feature_indices:
            feature_column_values = feature_data[:, feature_column_idx]

            # Quantile Binning - Only check quantile points instead of every unique value
            if len(feature_column_values) > self.num_quantile_bins:
                split_thresholds = np.percentile(feature_column_values, np.linspace(0, 100, self.num_quantile_bins))
            else:
                split_thresholds = np.unique(feature_column_values)

            for threshold_candidate in split_thresholds:
                information_gain = self._variance_reduction(target_data, feature_column_values, threshold_candidate)
                if information_gain > best_information_gain:
                    best_information_gain, best_feature_idx, best_split_threshold = information_gain, feature_column_idx, threshold_candidate

        return best_feature_idx, best_split_threshold

    def _information_gain(self, target_data, feature_column_values, split_threshold):
        parent_entropy = self._entropy(target_data)
        left_indices, right_indices = self._split(feature_column_values, split_threshold)
        if len(left_indices) == 0 or len(right_indices) == 0: 
            return 0
        n_total = len(target_data)
        n_left, n_right = len(left_indices), len(right_indices)
        entropy_left, entropy_right = self._entropy(target_data[left_indices]), self._entropy(target_data[right_indices])
        child_entropy = (n_left / n_total) * entropy_left + (n_right / n_total) * entropy_right
        return parent_entropy - child_entropy

    def _variance_reduction(self, target_data, feature_column_values, split_threshold):
        left_classification_mask = feature_column_values <= split_threshold
        right_classification_mask = ~left_classification_mask 
        
        if not np.any(left_classification_mask) or not np.any(right_classification_mask): 
            return 0
        
        n_total = len(target_data)
        n_left, n_right = np.sum(left_classification_mask), np.sum(right_classification_mask)
        # Vectorized variance reduction calculation
        return np.var(target_data) - ((n_left/n_total) * np.var(target_data[left_classification_mask]) + (n_right/n_total) * np.var(target_data[right_classification_mask]))

    def _split(self, feature_column_values, split_threshold):
        left_sample_indices = np.argwhere(feature_column_values <= split_threshold).flatten()
        right_sample_indices = np.argwhere(feature_column_values > split_threshold).flatten()
        return left_sample_indices, right_sample_indices

    def predict(self, feature_data):
        predictions_array = np.zeros(feature_data.shape[0])
        self._predict_vectorized(feature_data, np.arange(feature_data.shape[0]), self.root_node, predictions_array)
        return predictions_array

    def _predict_vectorized(self, feature_data, sample_indices, current_node, predictions_array):
        if current_node.value is not None:
            predictions_array[sample_indices] = current_node.value
            return

        feature_values_subset = feature_data[sample_indices, current_node.feature]
        left_sample_mask = feature_values_subset <= current_node.threshold
        
        if np.any(left_sample_mask):
            self._predict_vectorized(feature_data, sample_indices[left_sample_mask], current_node.left, predictions_array)
        if np.any(~left_sample_mask):
            self._predict_vectorized(feature_data, sample_indices[~left_sample_mask], current_node.right, predictions_array)

    def _traverse_tree(self, sample_point, current_node):
        if current_node.value is not None: 
            return current_node.value
        if sample_point[current_node.feature] <= current_node.threshold:
            return self._traverse_tree(sample_point, current_node.left)
        return self._traverse_tree(sample_point, current_node.right)

def run_decision_tree(feature_matrix, target_vector):
    # This remains a classifier for the "Decision Tree" menu option in app.py
    # But note: The internal logic above now supports XGBoost's floats
    model_instance = CustomDecisionTree()
    model_instance.fit(feature_matrix.values, target_vector.values)
    probability_predictions = model_instance.predict(feature_matrix.values)
    # Binary threshold for the standalone tree option
    binary_class_predictions = np.array([1 if pred_val > 0.5 else 0 for pred_val in probability_predictions])
    return binary_class_predictions, probability_predictions