import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate_param=0.01, max_iterations=1000):
        self.learning_rate_param = learning_rate_param
        self.max_iterations = max_iterations
        self.coefficient_weights = None
        self.intercept_bias = None

    def sigmoid_activation(self, linear_combination):
        linear_combination = np.clip(linear_combination, -500, 500)
        linear_combination = np.array(linear_combination, dtype=float)
        return 1 / (1 + np.exp(-linear_combination))

    def fit(self, feature_matrix, target_vector):
        # 1. Force conversion to NumPy float64 arrays to avoid 'O' (Object) dtypes
        feature_matrix = np.array(feature_matrix, dtype=np.float64)
        target_vector = np.array(target_vector, dtype=np.float64).reshape(-1, 1) # Ensure y is a column vector

        sample_count, feature_count = feature_matrix.shape

        # 2. Initialize weights explicitly as float64
        self.coefficient_weights = np.zeros((feature_count, 1), dtype=np.float64)
        self.intercept_bias = 0.0

        for iteration_step in range(self.max_iterations):
            linear_model_output = np.dot(feature_matrix, self.coefficient_weights) + self.intercept_bias
            predicted_probability = self.sigmoid_activation(linear_model_output)

            weight_gradient = (1 / sample_count) * np.dot(feature_matrix.T, (predicted_probability - target_vector))
            bias_gradient = (1 / sample_count) * np.sum(predicted_probability - target_vector)

            self.coefficient_weights -= self.learning_rate_param * weight_gradient
            self.intercept_bias -= self.learning_rate_param * bias_gradient

    def predict_probability(self, feature_matrix):
        linear_model_output = np.dot(feature_matrix, self.coefficient_weights) + self.intercept_bias
        return self.sigmoid_activation(linear_model_output)

    def predict_class(self, feature_matrix):
        return [1 if probability_score > 0.5 else 0 for probability_score in self.predict_probability(feature_matrix)]

def run_logistic_regression(feature_matrix, target_vector):
    model_instance = CustomLogisticRegression()
    model_instance.fit(feature_matrix.values, target_vector.values)
    return model_instance.predict_class(feature_matrix.values), model_instance.predict_probability(feature_matrix.values)