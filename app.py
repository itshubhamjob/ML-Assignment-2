import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import metrics from sklearn for the evaluation dashboard
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix)

# Import the custom "from scratch" functions
from models.logistic_regression import run_logistic_regression
from models.decision_tree import run_decision_tree
from models.knn import run_knn
from models.naive_bayes import run_naive_bayes
from models.random_forest import run_random_forest
from models.xg_boost_model import run_xgboost

st.set_page_config(page_title="Churn Classification (From Scratch)", layout="wide")

st.title("📊 Customer Churn Classification - Custom Implementations")
st.markdown("This application uses machine learning models built from scratch without scikit-learn's estimator classes.")

# Sidebar Configuration
st.sidebar.header("Upload & Model Settings")
csv_input_file = st.sidebar.file_uploader("Upload Test CSV (with 'Churn' column)", type="csv")
selected_classifier = st.sidebar.selectbox("Select ML Model", 
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])

# Helper function to process data
def preprocess_data(dataset_df):
    # Drop ID and handle the target
    if 'customerID' in dataset_df.columns:
        dataset_df = dataset_df.drop(columns=['customerID'])
    
    # Map Churn to binary
    if 'Churn' in dataset_df.columns:
        dataset_df['Churn'] = dataset_df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Simple encoding for categorical variables
    dataset_df = pd.get_dummies(dataset_df)
    
    # Fill missing values if any
    dataset_df = dataset_df.fillna(0)
    return dataset_df

if csv_input_file:
    input_data = pd.read_csv(csv_input_file)
    st.write("### Data Preview", input_data.head(5))
    
    processed_input = preprocess_data(input_data)
    
    if 'Churn' in processed_input.columns:
        feature_matrix = processed_input.drop(columns=['Churn'])
        target_labels = processed_input['Churn']
        
        # Mapping model selection to the imported scratch functions
        classifier_functions = {
            "Logistic Regression": run_logistic_regression,
            "Decision Tree": run_decision_tree,
            "KNN": run_knn,
            "Naive Bayes": run_naive_bayes,
            "Random Forest": run_random_forest,
            "XGBoost": run_xgboost
        }
        
        with st.spinner(f"Training and evaluating {selected_classifier}..."):
            # Execute the custom model
            # These return (output_predictions, output_probabilities) or (output_predictions, output_predictions)
            output_predictions, output_probabilities = classifier_functions[selected_classifier](feature_matrix, target_labels)
        
        # Calculation of Metrics
        performance_metrics = {
            "Accuracy": accuracy_score(target_labels, output_predictions),
            "AUC Score": roc_auc_score(target_labels, output_probabilities),
            "Precision": precision_score(target_labels, output_predictions, zero_division=0),
            "Recall": recall_score(target_labels, output_predictions, zero_division=0),
            "F1 Score": f1_score(target_labels, output_predictions, zero_division=0),
            "MCC Score": matthews_corrcoef(target_labels, target_labels) if len(np.unique(target_labels)) > 1 else 0
        }

        # Dashboard Layout
        st.divider()
        st.subheader(f"Evaluation Metrics: {selected_classifier}")
        
        metric_columns = st.columns(6)
        for idx, (metric_label, metric_value) in enumerate(performance_metrics.items()):
            metric_columns[idx].metric(metric_label, f"{metric_value:.4f}")

        # Confusion Matrix and Report
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("#### Confusion Matrix")
            confusion_mat = confusion_matrix(target_labels, output_predictions)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            st.pyplot(fig)
            
        with col2:
            st.write("#### Prediction Distribution")
            distribution_df = pd.DataFrame({'Actual': target_labels, 'Predicted': output_predictions})
            st.bar_chart(distribution_df.apply(pd.Series.value_counts))
            
    else:
        st.error("Error: The dataset must contain a 'Churn' column for evaluation.")

else:
    st.info("Please upload a CSV file to begin.")