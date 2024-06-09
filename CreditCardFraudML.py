# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 22:14:11 2024

@author: Apple iMac
"""
import sys
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

# Redirect standard output to a file
with open("entire_output.txt", "w") as f:
    original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = f  # Change the standard output to the file we created

    # Load data
    df = pd.read_excel('CreditCardDataSubset.xlsx', sheet_name='credit_card')

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Drop rows with any missing values
    df = df.dropna()

    # Identify the dependent variable (last column)
    dependent_var = df.columns[-1]

    df = pd.get_dummies(df, drop_first=True)

    # Convert boolean columns to uint8
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype('uint8')

    # Separate dependent and independent variables
    X = df.drop(dependent_var, axis=1)
    y = df[dependent_var]

    # Add a constant to the independent variables (for the intercept)
    X = sm.add_constant(X)

    # Perform regression
    model_ols = sm.OLS(y, X).fit()

    # Get the summary of the regression
    summary = model_ols.summary()
    print(summary)

    # Save summary to a file
    with open("model_summary.txt", "w") as summary_file:
        summary_file.write(summary.as_text())

    # Extract p-values and coefficients
    p_values = model_ols.pvalues.round(4)
    coefficients = model_ols.params.round(4)

    # Create a DataFrame for the results
    results = pd.DataFrame({
        'Coefficient': coefficients,
        'p-value': p_values
    })

    # Filter variables with p-value less than 0.05
    significant_results = results[results['p-value'] < 0.05]

    # Print the table
    print("Significant variables with p-value < 0.05:")
    print(significant_results)

    # Print each row individually where p-value < 0.05 in tabular format
    print("\nDetailed significant variables with p-value < 0.05:")
    print(f"{'Variable':<20}{'Coefficient':<20}{'p-value':<20}")
    print("-" * 60)
    for index, row in significant_results.iterrows():
        print(f"{index:<20}{row['Coefficient']:<20}{row['p-value']:<20}")

    # Identify significant independent variables
    significant_vars = significant_results.index
    #significant_vars = significant_vars.drop('const')  # Remove constant if present

    # Create a new DataFrame with significant independent variables
    X_significant = df[significant_vars]

    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_significant, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }

    # Train models and collect performance metrics
    metrics = {}
    roc_curves = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        metrics[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_prob)
        }

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_curves[name] = (fpr, tpr)

    # Print all metrics in one place
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df)

    # Plot all ROC curves in one graph
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr) in roc_curves.items():
        plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {metrics[name]["ROC-AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall curves for all models
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{name} (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    # Plot confusion matrices for all models
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    axes = axes.flatten()
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

    # Reset standard output to its original value
    sys.stdout = original_stdout

# Notify user
print("The entire output has been saved to 'entire_output.txt'.")
