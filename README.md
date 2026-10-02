# E-commerce Conversion Analysis

An end-to-end machine learning web application that predicts the
probability of an online shopping visitor making a purchase based
on session behavior.

## Project Overview

The application analyzes visitor session information such as:

- Product-related pages visited
- Product browsing duration
- Administrative activity
- Informational activity
- Bounce rate
- Exit rate
- Page values
- Visitor type
- Month
- Traffic type
- Weekend behavior

A Random Forest classifier generates a predicted purchase probability.

The application also uses SHAP to explain which features influenced
each individual prediction.

## Machine Learning Workflow

The project follows this workflow:

1. Load the Online Shoppers Purchasing Intention dataset
2. Inspect and clean the data
3. Remove duplicate records
4. Separate features and target
5. Identify numerical and categorical features
6. Build a preprocessing pipeline
7. Train classification models
8. Compare model performance
9. Tune the Random Forest using GridSearchCV
10. Evaluate the final model
11. Save the complete preprocessing + model pipeline
12. Deploy the model using Flask
13. Generate individual prediction explanations using SHAP

## Final Model

The final model is a Random Forest classifier.

Selected parameters:

- `n_estimators = 300`
- `max_depth = None`
- `min_samples_split = 2`
- `min_samples_leaf = 1`

## Model Performance

Final test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 90.66% |
| ROC-AUC | 92.57% |
| Purchase Precision | 78% |
| Purchase Recall | 57% |
| Purchase F1-score | 65% |

Because the dataset is imbalanced, accuracy is not used as the
only evaluation metric. Precision, recall, F1-score and ROC-AUC
provide additional information about model performance.

## Explainable AI

SHAP is used to explain individual predictions.

The application displays:

- Positive contributors
- Negative contributors
- SHAP values
- Session-level behavioral signals

This makes the prediction more interpretable than showing only
a probability.

## Web Application

The Flask application provides:

- Visitor session input form
- Purchase probability prediction
- Purchase intent classification
- Session analysis
- SHAP explanation
- Business recommendations
- Model performance information
- Project information

## Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SHAP

### Web Application

- Flask
- HTML
- CSS
- Jinja2

### Development

- Jupyter Notebook
- VS Code

## Running the Project

Install the dependencies:

```bash
pip install -r requirements.txt