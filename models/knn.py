import numpy as np
from collections import Counter

class CustomKNN:
    def __init__(self, neighbor_count=3):
        self.neighbor_count = neighbor_count

    def fit(self, training_features, training_labels):
        self.training_feature_matrix = training_features
        self.training_label_vector = training_labels

    def predict(self, test_features):
        class_predictions = [self._predict_single_sample(sample_vector) for sample_vector in test_features]
        return np.array(class_predictions)

    def _predict_single_sample(self, query_sample):
        distance_list = [np.sqrt(np.sum((query_sample - train_sample)**2)) for train_sample in self.training_feature_matrix]
        nearest_indices = np.argsort(distance_list)[:self.neighbor_count]
        nearest_neighbor_labels = [self.training_label_vector[neighbor_idx] for neighbor_idx in nearest_indices]
        most_frequent = Counter(nearest_neighbor_labels).most_common(1)
        return most_frequent[0][0]

def run_knn(feature_matrix, target_vector):
    model_instance = CustomKNN(neighbor_count=5)
    model_instance.fit(feature_matrix.values, target_vector.values)
    class_predictions = model_instance.predict(feature_matrix.values)
    return class_predictions, class_predictions