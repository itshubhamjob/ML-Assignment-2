**ML Assignment 2
**

A) Problem Statement: The aim of this assignment is to apply and compare multiple machine learning classification models on a chosen dataset that has at least 12 features and 500 records. Six algorithms—Logistic Regression, Decision Tree, K Nearest Neighbour, Naive Bayes, Random Forest, and XGBoost—must be implemented on the same dataset. Their performance will be evaluated using standard metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).
Alongside model building, the task requires developing an interactive Streamlit application where users can upload test data, select models, and view evaluation results with metrics and confusion matrices. The complete workflow, including source code, requirements, and documentation, should be maintained in a GitHub repository and deployed on Streamlit Community Cloud. This exercise is designed to give practical exposure to the end to end ML pipeline, covering model implementation, performance benchmarking, UI design, and deployment.
	B) Dataset Description: The dataset used in this assignment is the Telco Customer Churn dataset, which contains information about telecom customers and whether they have discontinued their service (churn). It includes 1,500 customer records with 21 features covering demographic details (such as gender, senior citizen status, partner, dependents), service-related attributes (phone service, internet service type, online security, streaming services, contract type), billing information (monthly charges, total charges, payment method), and the target variable Churn (Yes/No).
This dataset is suitable for classification tasks as it provides a mix of categorical and numerical features, with sufficient size and diversity to train and evaluate multiple machine learning models. The goal is to predict customer churn based on these attributes, making it a practical example of a real world business problem in customer retention and telecom analytics.


	C) Models Used: Here we have made a comparison with the evaluation metrics calculated for all the 6 models as below.
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.5384	0.8226	0.3598	0.9596	0.5234	1
Decision Tree	0.8299	0.8838	0.7311	0.5631	0.6362	1
KNN	0.8486	0.7911	0.7341	0.6692	0.7001	1
Naive Bayes	0.992	0.9946	0.9706	1	0.9851	1
Random Forest (Ensemble)	0.9093	0.8687	0.8611	0.7828	0.8201	1
XGBoost (Ensemble)	0.7358	0.8675	0	0	0	1

ML Model Name	Observation about model performance
**Logistic Regression**:-	This model achieved relatively low accuracy (0.5384) but an excellent recall (0.9596), meaning it correctly identified most churn cases. However, precision was poor (0.3598), indicating many false positives. The high AUC (0.8226) shows decent discriminatory power, but overall, the model is imbalanced—better at catching churn but not reliable in avoiding misclassification of non churn customers.
**Decision Tree**:-	The decision tree performed well with accuracy (0.8299) and a strong AUC (0.8838). Precision (0.7311) was higher than recall (0.5631), suggesting the model is conservative in predicting churn but more accurate when it does. The F1 score (0.6362) reflects this trade off. Overall, it provides solid performance but tends to miss some churn cases.
**KNN**:-	KNN showed strong accuracy (0.8486) and balanced precision (0.7341) and recall (0.6692). Its F1 score (0.7001) indicates good overall balance between false positives and false negatives. However, the AUC (0.7911) was lower compared to other models, meaning its ability to separate churn vs. non churn is weaker despite high accuracy.
**Naive Bayes**:-	This model delivered outstanding results with extremely high accuracy (0.992), perfect recall (1.0), and very high precision (0.9706). The F1 score (0.9851) and AUC (0.9946) confirm near perfect classification. It is the best performer among all models, showing excellent generalization and balance.
**Random Forest (Ensemble)**:-	Random Forest achieved high accuracy (0.9093) and strong precision (0.8611) with recall (0.7828). The F1 score (0.8201) shows balanced performance, and the AUC (0.8687) indicates good discriminatory ability. This ensemble model is robust and reliable, performing consistently across metrics.
**XGBoost (Ensemble)**:-	Surprisingly, XGBoost underperformed here, with moderate accuracy (0.7358) but zero precision, recall, and F1 score. This suggests the model failed to correctly classify churn cases, possibly due to poor parameter tuning or data imbalance. Despite a decent AUC (0.8675), the practical predictive performance was ineffective





