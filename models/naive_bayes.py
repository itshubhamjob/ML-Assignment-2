import numpy as np

class CustomGaussianNB:
    def fit(self, feature_matrix, target_vector):
        sample_count, feature_count = feature_matrix.shape
        self.class_labels = np.unique(target_vector)
        class_count = len(self.class_labels)

        self.feature_means = np.zeros((class_count, feature_count))
        self.feature_variances = np.zeros((class_count, feature_count))
        self.class_priors = np.zeros(class_count)

        for label_idx, class_label in enumerate(self.class_labels):
            features_for_class = feature_matrix[target_vector == class_label]
            self.feature_means[label_idx, :] = features_for_class.mean(axis=0)
            self.feature_variances[label_idx, :] = features_for_class.var(axis=0)
            self.class_priors[label_idx] = features_for_class.shape[0] / float(sample_count)

    def predict(self, feature_matrix):
        return np.array([self._predict_single_sample(sample_vector) for sample_vector in feature_matrix])

    def _predict_single_sample(self, feature_vector):
        posterior_probabilities = []
        for label_idx, class_label in enumerate(self.class_labels):
            prior_prob = np.log(self.class_priors[label_idx])
            conditional_likelihood = np.sum(np.log(self._gaussian_pdf(label_idx, feature_vector)))
            posterior_prob = prior_prob + conditional_likelihood
            posterior_probabilities.append(posterior_prob)
        return self.class_labels[np.argmax(posterior_probabilities)]

    def _gaussian_pdf(self, class_idx, feature_vector):
        feature_mean = self.feature_means[class_idx]
        feature_var = self.feature_variances[class_idx]

        epsilon_smoothing = 1e-9
        feature_var = feature_var + epsilon_smoothing

        exponent_term = -((feature_vector.astype(float) - feature_mean)**2) / (2 * feature_var)
        numerator_term = np.exp(exponent_term)
        denominator_term = np.sqrt(2 * np.pi * feature_var)
        return numerator_term / denominator_term

def run_naive_bayes(feature_matrix, target_vector):
    model_instance = CustomGaussianNB()
    model_instance.fit(feature_matrix.values, target_vector.values)
    class_predictions = model_instance.predict(feature_matrix.values)
    return class_predictions, class_predictions