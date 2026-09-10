# Factory Reallocation & Shipping Optimization Recommendation System

## Nassau Candy Distributor — E-Commerce Analytics

A machine learning based decision-support system for identifying slow-shipment risk and evaluating factory reallocation scenarios.

## Project Overview

This project analyzes historical order and shipping data and develops a machine learning workflow to predict slow shipments and evaluate alternative factory assignments.

The project covers:
- Data understanding and validation
- Delivery lead-time analysis
- Product-to-factory mapping
- Customer geography analysis
- Factory-region analysis
- Shipping-mode analysis
- Data leakage auditing
- Machine learning classification
- Model evaluation and selection
- Factory reallocation scenario analysis
- Model explainability
- Business impact analysis
- Final project integrity validation

## Dataset

- 10,194 records
- 18 original columns
- 15 products
- 5,044 customers
- 8,549 unique orders
- 3 divisions
- 4 regions
- 4 shipping modes

## Machine Learning Target

The target variable is `Slow_Shipment`.

A shipment is classified as slow when its delivery lead time is greater than the training-set Q3 threshold.

## ML Features

The model uses 15 input features:

1. Product Name
2. Division
3. Country/Region
4. State/Province
5. Region
6. Factory
7. Ship Mode
8. Units
9. Sales
10. Cost
11. Gross Profit
12. Order_Year
13. Order_Month
14. Order_Quarter
15. Order_DayOfWeek

Direct leakage variables such as Ship Date, Delivery_Days and Slow_Shipment are excluded from model inputs.

## Train/Test Methodology

Because multiple records can belong to the same order, an order-level group-aware train/test split was used.

- Training records: 8,132
- Testing records: 2,062
- Training orders: 6,839
- Testing orders: 1,710
- Order overlap: 0

## Models Evaluated

Three classification models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

## Model Performance

### Logistic Regression

- Accuracy: 0.7978
- Precision: 0.5070
- Recall: 0.9322
- F1 Score: 0.6568
- ROC-AUC: 0.8863

### Random Forest

- Accuracy: 0.8235
- Precision: 0.5457
- Recall: 0.8925
- F1 Score: 0.6773
- ROC-AUC: 0.9013

### Gradient Boosting

- Accuracy: 0.8346
- Precision: 0.5868
- Recall: 0.6869
- F1 Score: 0.6329
- ROC-AUC: 0.9043

## Selected Model

The final operational model is a Random Forest Classifier.

Selection prioritized:
1. F1 Score
2. Recall
3. ROC-AUC

The trained model pipeline is stored in:

`models/nassau_slow_shipment_model.joblib`

## Factory Reallocation Analysis

The system evaluates alternative factory assignments for each product and compares predicted slow-shipment probabilities.

The analysis includes:
- Current factory
- Scenario factory
- Current slow probability
- Predicted slow probability
- Probability improvement
- ML signal
- Final operational action

These recommendations are ML-based counterfactual screening results and are not causal estimates.

## Factory Coordinates

Verified factory coordinates were not available in the supplied project data.

Therefore, the project does not invent factory coordinates or claim transportation-cost savings.

## Model Explainability

The project includes:
- Random Forest feature importance
- Test-set permutation importance using ROC-AUC

Permutation importance evaluates the change in model performance when an input feature is shuffled on the held-out test set.

## Project Structure

```text
Nassau_Candy_Optimization/
|-- data/
|-- models/
|-- notebooks/
|-- results/
|-- .gitignore
|-- README.md
|-- requirements.txt
```

## Main Outputs

- Factory reallocation scenarios
- Factory reallocation recommendations
- Final factory reallocation recommendations
- Random Forest feature importance
- Test-set permutation importance
- Final business impact summary
- Final business priority table
- Final project file inventory

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook
- VS Code
- Git
- GitHub

## Reproducibility

Install dependencies with:

`pip install -r requirements.txt`

Then open:

`notebooks/01_data_understanding.ipynb`

## Business Value

The system helps identify slow-shipment risk and supports investigation of products where alternative factory assignments produce lower predicted slow-shipment probabilities.

Operational recommendations should be validated against factory capacity, production feasibility, transportation costs, inventory constraints and controlled pilot testing before implementation.

## Important Limitation

Factory reallocation results are model-based scenarios, not guaranteed operational outcomes or causal effects.

Transportation-cost savings are not estimated because verified factory coordinates and transportation-cost data were unavailable.

## Author

**Krishna Prajapati**

Machine Learning | AI & ML | Data Analytics
