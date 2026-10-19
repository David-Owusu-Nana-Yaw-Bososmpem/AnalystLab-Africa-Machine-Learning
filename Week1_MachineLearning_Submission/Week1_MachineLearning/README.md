# ShopSmart Retail Ltd. - Week 1 Machine Learning Project

## AnalystLab Africa Internship Programme
**Track:** Machine Learning  
**Mission:** ML01 - Building Your First Machine Learning Models

## Project Overview
This project develops baseline machine learning models using the ShopSmart Retail dataset. It covers data integration and preparation, regression, classification, model evaluation, feature importance, business analysis, model persistence, and a simple prediction script.

## Business Problems
1. **Regression:** Predict total order value (`Order_Total`).
2. **Classification:** Predict whether an order will be returned (`Return_Flag`).

## Dataset
The project integrates customer, product, order, store, return, employee, supplier, inventory, and order-item information. The final master dataset contains **50,000 unique orders** with no duplicate Order IDs. The observed return rate is **4.88%**.

## Models and Results

### Linear Regression
- MAE: 31,010.22
- RMSE: 41,268.26
- R²: 0.8830

The strongest variables in the fitted model include `Unit_Price` and `Quantity`, followed by several brand indicators.

### Logistic Regression
- Accuracy: 0.5430
- Precision: 0.0475
- Recall: 0.4385
- F1 Score: 0.0856

Returns are highly imbalanced: only 4.88% of orders were returned. The low precision and F1 score show that the baseline return model requires improvement before operational deployment.

## Business Insights
- Silver customers recorded the highest average order value (154,578.70), although the three customer segments were close.
- Customers aged 55-64 had the highest average order value (156,209.30).
- STR015 (ShopSmart Lagos, North) generated the highest total sales in the analysis: 317,657,047.
- Brand indicators were prominent among the largest logistic-regression coefficients for return prediction.
- The regression model is useful as a baseline for spending prediction, while the classification model should be treated cautiously because of class imbalance and weak positive-class precision.

## Project Structure
```text
Week1_MachineLearning/
├── ML01_Foundations.ipynb
├── data/
│   ├── master_dataset.csv
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── linear_regression.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
├── scripts/
│   └── predict.py
├── reports/
│   ├── ML_Report.pdf
│   └── Presentation.pptx
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run
1. Create and activate a Python virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Open `ML01_Foundations.ipynb` in Jupyter Notebook and run the cells in order.
4. To test the saved models interactively:
   `python scripts/predict.py`

## Important Note on Saved Models
The saved scikit-learn objects were produced with scikit-learn 1.6.1. For reliable loading, use the version pinned in `requirements.txt`.

## Limitations
- Linear and logistic regression impose relatively simple functional forms.
- Returns are strongly imbalanced.
- Historical patterns may change over time.
- Important behavioral or operational predictors may be absent.
- `Order_Total` is mathematically constructed from quantity, unit price, discount, and shipping cost, so strong regression performance partly reflects that relationship.
- Results are predictive associations and should not be interpreted as causal effects.
